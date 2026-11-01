#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <tuple>

namespace {
constexpr int MAX_K = 128;
constexpr float COS_10 = 0.984807753012208f;

struct Member {
    int slot, id, axis;
    float tau, alpha;
    float3 m, n;
};

__device__ inline float dot3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ inline float3 sub3(float3 a, float3 b) {
    return {a.x-b.x, a.y-b.y, a.z-b.z};
}
__device__ inline bool finite3(float3 x) {
    return isfinite(x.x) && isfinite(x.y) && isfinite(x.z);
}
__device__ inline float3 load3(const float* p) {
    return {p[0], p[1], p[2]};
}
__device__ inline float3 normal_axis(const float* r, int a) {
    return {r[a], r[3+a], r[6+a]};
}
__device__ inline float cov_entry(const float* r, const float* s, int i, int j) {
    float out = 0.f;
    for (int a=0; a<3; ++a) out += (r[3*i+a]*(s[a]*s[a]))*r[3*j+a];
    return out;
}
__device__ inline bool distinct_from(const Member& a, const Member& b,
                                     const float* rot, const float* scales) {
    if (a.m.x != b.m.x || a.m.y != b.m.y || a.m.z != b.m.z) return true;
    const float* ra = rot + static_cast<int64_t>(a.id)*9;
    const float* rb = rot + static_cast<int64_t>(b.id)*9;
    const float* sa = scales + static_cast<int64_t>(a.id)*3;
    const float* sb = scales + static_cast<int64_t>(b.id)*3;
    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j)
            if ((cov_entry(ra,sa,i,j)+cov_entry(ra,sa,j,i))*0.5f !=
                (cov_entry(rb,sb,i,j)+cov_entry(rb,sb,j,i))*0.5f) return true;
    return false;
}

__device__ inline void compatibility(const Member& a, const Member& b,
                                     float pixel_scale, bool& angle, bool& distance) {
    angle = fabsf(dot3(a.n,b.n)) >= COS_10;
    const float3 d = sub3(b.m,a.m);
    const float gap = fmaxf(fabsf(dot3(a.n,d)), fabsf(dot3(b.n,d)));
    distance = gap <= (a.tau+b.tau)*pixel_scale;
}

// Audit columns: int 0 invalid, 1 valid, 2 basic, 3 short, 4 grazing,
// 5 tau, 6 first size, 7 chosen group, 8 skipped groups, 9 skipped members,
// 10 selected size. Float: first alpha, skipped alpha, skipped tau min/max.
// Flags: candidate, group switch, boundary angle/distance, no valid,
// low alpha, footprint failure, insufficient distinct geometry.
__global__ void selectCUDA(
    int Q, int K, int N, const float* centers, const float* rot,
    const float* scales, const bool* primitive_valid,
    const float* xy, const int* ids, const float* alpha,
    const int* count, const bool* overflow,
    float fx, float fy, float cx, float cy,
    int* chosen_slots, int* chosen_axes, int* audit_i,
    float* audit_f, bool* audit_b) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Q) return;
    const int64_t base = static_cast<int64_t>(q)*K;
    int* ai = audit_i + static_cast<int64_t>(q)*11;
    float* af = audit_f + static_cast<int64_t>(q)*4;
    bool* ab = audit_b + static_cast<int64_t>(q)*8;
    ai[7] = -1;
    const float px=xy[static_cast<int64_t>(q)*2], py=xy[static_cast<int64_t>(q)*2+1];
    const bool query_valid=isfinite(px) && isfinite(py);
    const float3 v = query_valid ? float3{(px-cx)/fx,(py-cy)/fy,1.f} : float3{0.f,0.f,1.f};
    const float vnorm=sqrtf(dot3(v,v));
    const float pixel_scale=fmaxf(1.f/fx,1.f/fy);
    Member sorted[MAX_K];
    int nv=0;
    const int slots = min(max(count[q],0),K);
    for (int slot=0; slot<slots; ++slot) {
        const int id=ids[base+slot];
        if (id<0 || id>=N) continue;
        const float a=alpha[base+slot];
        const float3 m=load3(centers+static_cast<int64_t>(id)*3);
        const float3 s=load3(scales+static_cast<int64_t>(id)*3);
        const float* r=rot+static_cast<int64_t>(id)*9;
        bool rfinite=true;
        for (int j=0; j<9; ++j) rfinite &= isfinite(r[j]);
        const bool basic=primitive_valid[id] && query_valid && finite3(m) &&
                         finite3(s) && s.x>0 && s.y>0 && s.z>0 &&
                         rfinite && isfinite(a) && a>=0 && a<1;
        if (!basic) {++ai[0];++ai[2];continue;}
        int axis=0;
        if (s.y<s.x) axis=1;
        if (s.z<(&s.x)[axis]) axis=2;
        const float smallest=axis==0 ? s.x : axis==1 ? s.y : s.z;
        const float middle=axis==0 ? fminf(s.y,s.z) :
                           axis==1 ? fminf(s.x,s.z) : fminf(s.x,s.y);
        if (!(middle/smallest>1.001f)) {++ai[0];++ai[3];continue;}
        const float3 n=normal_axis(r,axis);
        const float inc=dot3(n,v);
        if (!(fabsf(inc)/vnorm>=.1f)) {++ai[0];++ai[4];continue;}
        const float tau=dot3(n,m)/inc;
        if (!(isfinite(tau) && tau>0.f)) {++ai[0];++ai[5];continue;}
        Member entry{slot,id,axis,tau,a,m,n};
        // Stable lexicographic (tau, original Gaussian ID, accepted slot).
        int at=nv;
        while (at>0 && (sorted[at-1].tau>tau ||
               (sorted[at-1].tau==tau && sorted[at-1].id>id))) {
            sorted[at]=sorted[at-1]; --at;
        }
        sorted[at]=entry;
        ++nv;
    }
    ai[1]=nv;
    ab[4]=nv==0;
    ab[7]=nv==0;
    int starts[MAX_K], ends[MAX_K], groups=0;
    if (nv) {
        starts[0]=0;
        groups=1;
        for (int p=1; p<nv; ++p) {
            bool compatible=true, any_angle=false, any_distance=false;
            for (int j=starts[groups-1]; j<p; ++j) {
                bool angle,distance;
                compatibility(sorted[j],sorted[p],pixel_scale,angle,distance);
                compatible &= angle && distance;
                any_angle |= !angle; any_distance |= !distance;
            }
            if (!compatible) {
                ends[groups-1]=p;
                if (groups==1) {ab[2]=any_angle;ab[3]=any_distance;}
                starts[groups++]=p;
            }
        }
        ends[groups-1]=nv;
    }
    const int first_size=nv ? ends[0] : 0;
    ai[6]=first_size;
    ab[1]=first_size<nv;
    float first_product=1.f;
    for (int j=0;j<first_size;++j) first_product*=1.f-sorted[j].alpha;
    af[0]=1.f-first_product;
    ab[5]=af[0]<.5f;

    int chosen=-1;
    for (int g=0; g<groups; ++g) {
        const int begin=starts[g], end=ends[g], len=end-begin;
        bool distinct=false;
        for (int j=begin+1;j<end;++j)
            distinct |= distinct_from(sorted[begin],sorted[j],rot,scales);
        if (g==0) ab[7]=!distinct;
        float product=1.f, numerator=0.f, denominator=0.f;
        for (int j=begin;j<end;++j) {
            const Member& m=sorted[j];
            product*=1.f-m.alpha;
            const float c=dot3(m.n,v), b=dot3(m.n,m.m);
            denominator+=c*c; numerator+=c*b;
        }
        const float z=numerator/denominator;
        bool footprint_fail=false;
        {
            const float3 point{z*v.x,z*v.y,z};
            for (int j=begin;j<end;++j) {
                const Member& m=sorted[j];
                const float3 d=sub3(point,m.m);
                const float* r=rot+static_cast<int64_t>(m.id)*9;
                const float* s=scales+static_cast<int64_t>(m.id)*3;
                float footprint=0.f;
                for (int a=0;a<3;++a) if (a!=m.axis) {
                    const float local=(r[a]*d.x+r[3+a]*d.y+r[6+a]*d.z)/s[a];
                    footprint+=local*local;
                }
                footprint_fail |= !isfinite(footprint) || footprint>4.f;
            }
        }
        if (g==0) ab[6]=footprint_fail;
        const bool qualified=len>=2 && distinct && 1.f-product>=.5f &&
             !footprint_fail && isfinite(denominator) && denominator>0 &&
             isfinite(z) && z>0;
        if (qualified) {chosen=g;break;}
    }
    const bool good=chosen>=0 && !overflow[q] && count[q]<=K;
    ab[0]=good;
    if (good) {
        ai[7]=chosen;
        ai[8]=chosen;
        ai[10]=ends[chosen]-starts[chosen];
        for (int j=starts[chosen]; j<ends[chosen]; ++j) {
            const int out=j-starts[chosen];
            chosen_slots[base+out]=sorted[j].slot;
            chosen_axes[base+out]=sorted[j].axis;
        }
        if (chosen>0) {
            ai[9]=starts[chosen];
            float product=1.f;
            for (int j=0;j<starts[chosen];++j) product*=1.f-sorted[j].alpha;
            af[1]=1.f-product;
            af[2]=sorted[0].tau;
            af[3]=sorted[starts[chosen]-1].tau;
        }
    }
}
} // namespace

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
SelectSupportGroupsCUDA(const torch::Tensor& centers,const torch::Tensor& rotations,
    const torch::Tensor& scales,const torch::Tensor& primitive_valid,
    const torch::Tensor& xy,const torch::Tensor& ids,const torch::Tensor& alpha,
    const torch::Tensor& count,const torch::Tensor& overflow,
    float fx,float fy,float cx,float cy) {
    TORCH_CHECK(centers.is_cuda() && centers.dim()==2 && centers.size(1)==3, "centers [N,3] CUDA");
    const auto device=centers.device();
    c10::cuda::CUDAGuard guard(device);
    const int64_t N=centers.size(0), Q=xy.size(0), K=ids.size(1);
    TORCH_CHECK(N>0 && N<=std::numeric_limits<int>::max() && Q>=0 &&
                Q<=std::numeric_limits<int>::max()-127 && K>0 && K<=MAX_K,
                "group selection dimensions exceed bounds");
    TORCH_CHECK(std::isfinite(fx) && std::isfinite(fy) && fx>0 && fy>0 &&
                std::isfinite(cx) && std::isfinite(cy), "invalid camera intrinsics");
    auto check=[&](const torch::Tensor& t,torch::ScalarType type,const char* name) {
        TORCH_CHECK(t.is_cuda() && t.device()==device && t.is_contiguous() &&
                    t.scalar_type()==type,name," device/dtype/layout mismatch");
    };
    check(centers,torch::kFloat32,"centers");
    check(rotations,torch::kFloat32,"rotations");
    check(scales,torch::kFloat32,"scales");
    check(primitive_valid,torch::kBool,"primitive_valid");
    check(xy,torch::kFloat32,"xy");
    check(ids,torch::kInt32,"ids");
    check(alpha,torch::kFloat32,"alpha");
    check(count,torch::kInt32,"count");
    check(overflow,torch::kBool,"overflow");
    TORCH_CHECK(rotations.dim()==3 && rotations.size(0)==N && rotations.size(1)==3 &&
                rotations.size(2)==3 && scales.dim()==2 && scales.size(0)==N &&
                scales.size(1)==3 && primitive_valid.dim()==1 &&
                primitive_valid.numel()==N && xy.dim()==2 && xy.size(1)==2 &&
                ids.dim()==2 && ids.size(0)==Q && alpha.sizes()==ids.sizes() &&
                count.dim()==1 && count.numel()==Q && overflow.dim()==1 &&
                overflow.numel()==Q, "group selection shape mismatch");
    auto chosen=torch::full({Q,K},-1,ids.options());
    auto axes=torch::full({Q,K},-1,ids.options());
    auto audit_i=torch::zeros({Q,11},ids.options());
    auto audit_f=torch::zeros({Q,4},alpha.options());
    auto audit_b=torch::zeros({Q,8},overflow.options());
    if (Q) {
        const auto stream=at::cuda::getCurrentCUDAStream(device.index());
        selectCUDA<<<(Q+127)/128,128,0,stream.stream()>>>(
            static_cast<int>(Q),static_cast<int>(K),static_cast<int>(N),
            centers.data_ptr<float>(),rotations.data_ptr<float>(),
            scales.data_ptr<float>(),primitive_valid.data_ptr<bool>(),
            xy.data_ptr<float>(),ids.data_ptr<int>(),alpha.data_ptr<float>(),
            count.data_ptr<int>(),overflow.data_ptr<bool>(),
            fx,fy,cx,cy,chosen.data_ptr<int>(),axes.data_ptr<int>(),
            audit_i.data_ptr<int>(),audit_f.data_ptr<float>(),audit_b.data_ptr<bool>());
        auto error=cudaGetLastError();
        TORCH_CHECK(error==cudaSuccess,"selectCUDA: ",cudaGetErrorString(error));
    }
    return {chosen,axes,audit_i,audit_f,audit_b};
}
