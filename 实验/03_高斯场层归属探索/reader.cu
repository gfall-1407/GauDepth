// Research-only readout derived from the preserved host forward (Inria LICENSE.md).
// Layout checked against rasterizer_impl.cu::fromChunk, NOT struct declaration order.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <limits>
#include <vector>

template<class T> T* take(char*& p, size_t n) {
    auto q=(reinterpret_cast<uintptr_t>(p)+127)&~uintptr_t(127);
    auto out=reinterpret_cast<T*>(q); p=reinterpret_cast<char*>(out+n); return out;
}
__global__ void trace(const uint2* ranges,const uint32_t* ids,const float2* xy,
 const float4* conic,const float4* plane,const float4* normals,const int* pixels,
 int W,int H,float fx,float fy,int N,int K,float* records,int* gids,int* counts,float* replay) {
 int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=N)return;
 int p=pixels[r],x=p%W,y=p/W;
 uint2 range=ranges[(y/16)*((W+15)/16)+x/16];
 float nx=(x-float(W-1)/2.f)/fx,ny=(y-float(H-1)/2.f)/fy;
 float rln=rnorm3df(nx,ny,1.f),T=1.f,D=0.f,M=0.f; int n=0;
 for(uint32_t j=range.x;j<range.y;++j){
  int id=ids[j];float2 d={xy[id].x-x,xy[id].y-y};float4 c=conic[id];
  float power=-0.5f*(c.x*d.x*d.x+c.z*d.y*d.y)-c.y*d.x*d.y;
  if(power>0.f)continue;
  float a=fminf(0.99f,c.w*expf(power));
  if(a<1.f/255.f)continue;
  float next=T*(1.f-a);if(next<0.0001f)break;
  float4 pl=plane[id],normal=normals[id];
  float t=pl.x*d.x+pl.y*d.y+pl.z,w=T*a;
  if(n<K){
   int k=(r*K+n)*8;gids[r*K+n]=id;
   records[k]=t*rln;records[k+1]=w;records[k+2]=a;records[k+3]=T;
   records[k+4]=normal.x;records[k+5]=normal.y;records[k+6]=normal.z;
   // The host's NaN fallback has zero ray plane and a plausible-looking normal.
   records[k+7]=(pl.x==0.f&&pl.y==0.f&&pl.z==0.f)?1.f:0.f;
  }
  D+=t*w;if(T>0.5f)M=t;T=next;++n;
 }
 counts[r]=n;replay[r*3]=n?D*rln/(1.f-T):0.f;
 replay[r*3+1]=M*rln;replay[r*3+2]=1.f-T;
}
std::vector<torch::Tensor> read_rays(torch::Tensor geom,torch::Tensor bin,
 torch::Tensor img,int P,int W,int H,double fx,double fy,torch::Tensor pixels,int K){
 TORCH_CHECK(geom.is_cuda()&&bin.is_cuda()&&img.is_cuda()&&pixels.is_cuda(),"CUDA required");
 TORCH_CHECK(geom.is_contiguous()&&bin.is_contiguous()&&img.is_contiguous()&&pixels.is_contiguous(),"Contiguous buffers required");
 TORCH_CHECK(geom.scalar_type()==torch::kUInt8&&bin.scalar_type()==torch::kUInt8&&img.scalar_type()==torch::kUInt8,"Byte buffers required");
 TORCH_CHECK(pixels.scalar_type()==torch::kInt32&&pixels.dim()==1,"int32 pixel vector required");
 TORCH_CHECK(geom.device()==bin.device()&&geom.device()==img.device()&&geom.device()==pixels.device(),"Same device required");
 int N=pixels.numel();TORCH_CHECK(N>0&&K>0&&P>0&&W>0&&H>0,"Invalid dimensions");
 TORCH_CHECK(pixels.min().item<int>()>=0&&pixels.max().item<int>()<W*H,"Pixel out of range");
 c10::cuda::CUDAGuard guard(geom.device());
 char* g=(char*)geom.data_ptr();take<float>(g,P);auto pl=take<float4>(g,P);
 auto normals=take<float4>(g,P);take<bool>(g,P*3);take<int>(g,P);auto xy=take<float2>(g,P);
 auto c=take<float4>(g,P);char* i=(char*)img.data_ptr();take<uint32_t>(i,W*H*2);
 auto ranges=take<uint2>(i,W*H);char* b=(char*)bin.data_ptr();auto ids=take<uint32_t>(b,0);
 auto rec=torch::empty({N,K,8},geom.options().dtype(torch::kFloat32));
 auto gids=torch::empty({N,K},geom.options().dtype(torch::kInt32));
 auto counts=torch::empty({N},geom.options().dtype(torch::kInt32));
 auto rep=torch::empty({N,3},geom.options().dtype(torch::kFloat32));
 trace<<<(N+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(ranges,ids,xy,c,pl,normals,
  pixels.data_ptr<int>(),W,H,fx,fy,N,K,rec.data_ptr<float>(),gids.data_ptr<int>(),counts.data_ptr<int>(),rep.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {rec,gids,counts,rep};
}
// Aggregate replay: O(HW) storage, no per-pixel/per-Gaussian dense record allocation.
__global__ void pixel_stats(const uint2* ranges,const uint32_t* ids,const float2* xy,
 const float4* conic,const uint32_t* host_last,int W,int H,int* counts,float* weights){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=W*H)return;
 int x=p%W,y=p/W;uint2 range=ranges[(y/16)*((W+15)/16)+x/16];
 int evaluated=0,power_rejected=0,alpha_rejected=0,stopped=0,accepted=0,last=0;
 float T=1.f,total=0.f,small=1.f,large=0.f;
 for(uint32_t j=range.x;j<range.y;++j){
  ++evaluated;int id=ids[j];float2 d={xy[id].x-x,xy[id].y-y};float4 c=conic[id];
  float power=-0.5f*(c.x*d.x*d.x+c.z*d.y*d.y)-c.y*d.x*d.y;
  if(power>0.f){++power_rejected;continue;}
  float a=fminf(.99f,c.w*expf(power));
  if(a<1.f/255.f){++alpha_rejected;continue;}
  float next=T*(1.f-a);if(next<.0001f){stopped=1;break;}
  float w=T*a;small=fminf(small,w);large=fmaxf(large,w);total+=w;
  T=next;++accepted;last=evaluated;
 }
 int* c=counts+p*8;c[0]=range.y-range.x;c[1]=evaluated;c[2]=power_rejected;c[3]=alpha_rejected;
 c[4]=stopped;c[5]=accepted;c[6]=last;c[7]=host_last[p];
 float* w=weights+p*4;w[0]=accepted?small:nanf("");w[1]=accepted?large:nanf("");w[2]=total;w[3]=T;
}

std::vector<torch::Tensor> read_pixel_stats(torch::Tensor geom,torch::Tensor bin,
 torch::Tensor img,int P,int W,int H){
 TORCH_CHECK(geom.is_cuda()&&bin.is_cuda()&&img.is_cuda(),"CUDA required");
 TORCH_CHECK(geom.is_contiguous()&&bin.is_contiguous()&&img.is_contiguous(),"Contiguous buffers required");
 TORCH_CHECK(geom.scalar_type()==torch::kUInt8&&bin.scalar_type()==torch::kUInt8&&img.scalar_type()==torch::kUInt8,"Byte buffers required");
 TORCH_CHECK(geom.device()==bin.device()&&geom.device()==img.device(),"Same device required");
 TORCH_CHECK(P>0&&W>0&&H>0,"Invalid dimensions");c10::cuda::CUDAGuard guard(geom.device());
 char* g=(char*)geom.data_ptr();take<float>(g,P);take<float4>(g,P);take<float4>(g,P);
 take<bool>(g,P*3);take<int>(g,P);auto xy=take<float2>(g,P);auto conic=take<float4>(g,P);
 char* i=(char*)img.data_ptr();auto host_last=take<uint32_t>(i,W*H*2);auto ranges=take<uint2>(i,W*H);
 char* b=(char*)bin.data_ptr();auto ids=take<uint32_t>(b,0);
 auto counts=torch::empty({H,W,8},geom.options().dtype(torch::kInt32));
 auto weights=torch::empty({H,W,4},geom.options().dtype(torch::kFloat32));
 pixel_stats<<<(W*H+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(ranges,ids,xy,conic,host_last,W,H,counts.data_ptr<int>(),weights.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {counts,weights};
}
// Derivative of the fixed weighted Huber objective. These are the saved original
// accepted weights; neither transmittance nor a new compositing pass is used.
__device__ double huber_derivative(const float* records,int count,double d,double h){
 double sum=0.0,correction=0.0;
 for(int j=0;j<count;++j){
  // Divide the derivative by its common positive h. Saturated terms then
  // multiply by exactly +/-1, preserving balanced zero intervals across units.
  double residual=(d-static_cast<double>(records[j*8]))/h;
  double clipped=fmin(1.0,fmax(-1.0,residual));
  double term=__dmul_rn(static_cast<double>(records[j*8+1]),clipped);
  double next=__dadd_rn(sum,term);
  // Neumaier compensation follows the CPU math.fsum definition closely, including
  // balanced separated supports whose minimizer is an interval rather than a point.
  double error=(fabs(sum)>=fabs(term))?
   __dadd_rn(__dsub_rn(sum,next),term):__dadd_rn(__dsub_rn(term,next),sum);
  correction=__dadd_rn(correction,error);sum=next;
 }
 return __dadd_rn(sum,correction);
}

__global__ void huber_readout(const float* records,const int* counts,int N,int K,
 double h,double* depth,int* status,double* width){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=N)return;
 int count=counts[r];
 if(count==0){depth[r]=0.0;status[r]=0;width[r]=0.0;return;}
 const float* ray=records+static_cast<int64_t>(r)*K*8;
 double lo=static_cast<double>(ray[0]),hi=lo;
 for(int j=0;j<count;++j){
  double z=static_cast<double>(ray[j*8]),w=static_cast<double>(ray[j*8+1]);
  if(!isfinite(z)||!isfinite(w)||!(w>0.0)){
   depth[r]=nan("");status[r]=2;width[r]=nan("");return;
  }
  lo=fmin(lo,z);hi=fmax(hi,z);
 }
 double left_lo=lo,left_hi=hi,right_lo=lo,right_hi=hi;
 for(int it=0;it<64;++it){
  double mid=left_lo+(left_hi-left_lo)*0.5;
  if(mid==left_lo||mid==left_hi)break;
  if(huber_derivative(ray,count,mid,h)>=0.0)left_hi=mid;else left_lo=mid;
 }
 for(int it=0;it<64;++it){
  double mid=right_lo+(right_hi-right_lo)*0.5;
  if(mid==right_lo||mid==right_hi)break;
  if(huber_derivative(ray,count,mid,h)<=0.0)right_lo=mid;else right_hi=mid;
 }
 double left=left_hi,right=right_lo;
 if(right<left){double temp=left;left=right;right=temp;}
 depth[r]=left+(right-left)*0.5;status[r]=1;width[r]=right-left;
}

// Return [float64 depth[N], int32 status[N], float64 minimizer_width[N]].
// Status: 0=empty (depth and width are zero), 1=finite nonempty, 2=invalid member.
// Overflowed or negative counts are a hard API error, never a truncated estimate.
std::vector<torch::Tensor> read_huber(torch::Tensor records,torch::Tensor counts,double h_scene){
 TORCH_CHECK(records.is_cuda()&&counts.is_cuda(),"CUDA required");
 TORCH_CHECK(records.is_contiguous()&&counts.is_contiguous(),"Contiguous records/counts required");
 TORCH_CHECK(records.device()==counts.device(),"Same device required");
 TORCH_CHECK(records.scalar_type()==torch::kFloat32&&records.dim()==3&&records.size(2)==8,
  "records must be float32 [N,K,8]");
 TORCH_CHECK(counts.scalar_type()==torch::kInt32&&counts.dim()==1&&counts.size(0)==records.size(0),
  "counts must be int32 [N]");
 TORCH_CHECK(std::isfinite(h_scene)&&h_scene>0.0,"h_scene must be finite and positive");
 TORCH_CHECK(records.size(0)<=std::numeric_limits<int>::max()&&records.size(1)>0&&
  records.size(1)<=std::numeric_limits<int>::max(),"Invalid N or K");
 c10::cuda::CUDAGuard guard(records.device());
 int N=static_cast<int>(records.size(0)),K=static_cast<int>(records.size(1));
 if(N){
  TORCH_CHECK(counts.min().item<int>()>=0&&counts.max().item<int>()<=K,
   "counts outside [0,K]; retry read_rays with enough capacity before read_huber");
 }
 auto depth=torch::empty({N},records.options().dtype(torch::kFloat64));
 auto status=torch::empty({N},records.options().dtype(torch::kInt32));
 auto width=torch::empty({N},records.options().dtype(torch::kFloat64));
 if(N){
  huber_readout<<<(N+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
   records.data_ptr<float>(),counts.data_ptr<int>(),N,K,h_scene,
   depth.data_ptr<double>(),status.data_ptr<int>(),width.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
 return {depth,status,width};
}

// Separate gap-Huber API; the original reader/Huber contracts remain unchanged.
// A proposal-support gap is not physical empty space. Original render weights
// are used as objective weights, not interpreted as physical probabilities.
__global__ void gap_huber_readout(const float* records,const int* counts,
 const float* native_expected,int N,int K,double h,double* selected,
 int* status,double* min_gap){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=N)return;
 double native=static_cast<double>(native_expected[r]);
 bool native_valid=isfinite(native)&&native>0.0;
 selected[r]=native_valid?native:nan("");status[r]=2;min_gap[r]=nan("");
 int count=counts[r];
 if(count==0){
  // A positive native value with an empty replay is unknown, not a valid empty ray.
  if(isfinite(native)&&native==0.0){selected[r]=native;status[r]=0;}
  return;
 }
 if(!native_valid)return;
 const float* ray=records+static_cast<int64_t>(r)*K*8;
 double total=0.0,numerator=0.0,lo=INFINITY,hi=-INFINITY;
 for(int j=0;j<count;++j){
  double z=static_cast<double>(ray[j*8]),w=static_cast<double>(ray[j*8+1]);
  // Match the CPU gap candidate: zero weight is allowed; invalid members are
  // never silently filtered, including a nonpositive/nonfinite zero-weight depth.
  if(!isfinite(z)||!(z>0.0)||!isfinite(w)||w<0.0)return;
  total=__dadd_rn(total,w);
  numerator=__dadd_rn(numerator,__dmul_rn(w,z));
  if(w>0.0){lo=fmin(lo,z);hi=fmax(hi,z);}
 }
 if(!isfinite(total)||!(total>0.0)||!isfinite(numerator))return;
 double expected=numerator/total;
 if(!isfinite(expected)||!(expected>0.0))return;
 double gap=INFINITY;
 for(int j=0;j<count;++j){
  if(ray[j*8+1]>0.0f)
   gap=fmin(gap,fabs(__dsub_rn(static_cast<double>(ray[j*8]),expected)));
 }
 min_gap[r]=gap; // Camera-z scene units, identical to h_scene (not mm unless scaled).
 if(gap<=h){
  // Preserve the supplied original float32 expected exactly after float64
  // promotion and eventual float32 storage. E64 above is only the gate center.
  selected[r]=native;status[r]=1;return;
 }
 // Only gap rays enter either derivative loop; no all-ray Huber precomputation.
 double left_lo=lo,left_hi=hi,right_lo=lo,right_hi=hi;
 for(int it=0;it<80;++it){
  double mid=left_lo+(left_hi-left_lo)*0.5;
  if(mid==left_lo||mid==left_hi)break;
  if(huber_derivative(ray,count,mid,h)>=0.0)left_hi=mid;else left_lo=mid;
 }
 for(int it=0;it<80;++it){
  double mid=right_lo+(right_hi-right_lo)*0.5;
  if(mid==right_lo||mid==right_hi)break;
  if(huber_derivative(ray,count,mid,h)<=0.0)right_lo=mid;else right_hi=mid;
 }
 double left=left_hi,right=right_lo;
 if(right<left){double temp=left;left=right;right=temp;}
 double value=left+(right-left)*0.5;
 if(!isfinite(value)||!(value>0.0))return;
 selected[r]=value;status[r]=3;
}

// Return [float64 selected[N], int32 status[N], float64 min_gap[N]].
// 0=empty with native zero; 1=no gap, exact supplied native value retained;
// 2=unknown (positive finite native fallback, otherwise NaN); 3=gap, Huber solved.
// min_gap is NaN for undefined/invalid input, otherwise the E64 support distance.
// The caller must supply original _C expected and verify its render/source
// provenance; tensor validation cannot authenticate the producer of a value.
std::vector<torch::Tensor> read_gap_huber(torch::Tensor records,torch::Tensor counts,
 torch::Tensor native_expected,double h_scene){
 TORCH_CHECK(records.is_cuda()&&counts.is_cuda()&&native_expected.is_cuda(),"CUDA required");
 TORCH_CHECK(records.is_contiguous()&&counts.is_contiguous()&&native_expected.is_contiguous(),
  "Contiguous records/counts/native_expected required");
 TORCH_CHECK(records.device()==counts.device()&&records.device()==native_expected.device(),"Same device required");
 TORCH_CHECK(records.scalar_type()==torch::kFloat32&&records.dim()==3&&records.size(2)==8,
  "records must be float32 [N,K,8]");
 TORCH_CHECK(counts.scalar_type()==torch::kInt32&&counts.dim()==1&&counts.size(0)==records.size(0),
  "counts must be int32 [N]");
 TORCH_CHECK(native_expected.scalar_type()==torch::kFloat32&&native_expected.dim()==1&&
  native_expected.size(0)==records.size(0),"native_expected must be float32 [N]");
 TORCH_CHECK(std::isfinite(h_scene)&&h_scene>0.0,"h_scene must be finite and positive");
 TORCH_CHECK(records.size(0)<=std::numeric_limits<int>::max()&&records.size(1)>0&&
  records.size(1)<=std::numeric_limits<int>::max(),"Invalid N or K");
 c10::cuda::CUDAGuard guard(records.device());
 int N=static_cast<int>(records.size(0)),K=static_cast<int>(records.size(1));
 if(N){
  TORCH_CHECK(counts.min().item<int>()>=0&&counts.max().item<int>()<=K,
   "counts outside [0,K]; retry read_rays with enough capacity before read_gap_huber");
 }
 auto selected=torch::empty({N},records.options().dtype(torch::kFloat64));
 auto status=torch::empty({N},records.options().dtype(torch::kInt32));
 auto gap=torch::empty({N},records.options().dtype(torch::kFloat64));
 if(N){
  gap_huber_readout<<<(N+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
   records.data_ptr<float>(),counts.data_ptr<int>(),native_expected.data_ptr<float>(),N,K,h_scene,
   selected.data_ptr<double>(),status.data_ptr<int>(),gap.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
 return {selected,status,gap};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("read_rays",&read_rays);m.def("read_pixel_stats",&read_pixel_stats);
 m.def("read_huber",&read_huber);
 m.def("read_gap_huber",&read_gap_huber);
}
