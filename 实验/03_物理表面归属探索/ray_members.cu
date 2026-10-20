// Research-only accepted-ray reader, derived from 01/ray_trace_layered.cu.
// Host buffer layout and equations: RaDe-GS cuda_rasterizer (Inria LICENSE.md).
// Read original depth/weight/normal/ID; never recompute sorted occlusion.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

template<class T> T* take(char*& p, size_t n) {
    auto q=(reinterpret_cast<uintptr_t>(p)+127)&~uintptr_t(127);
    auto out=reinterpret_cast<T*>(q);p=reinterpret_cast<char*>(out+n);return out;
}
__global__ void trace(const uint2* ranges,const uint32_t* ids,const float2* xy,
 const float4* conic,const float4* plane,const float4* normals,const int* pixels,
 int W,int H,float fx,float fy,int N,int K,float* records,int* member_ids,
 int* counts,float* replay) {
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=N)return;
 int p=pixels[r],x=p%W,y=p/W;uint2 range=ranges[(y/16)*((W+15)/16)+x/16];
 float nx=(x-float(W-1)/2.f)/fx,ny=(y-float(H-1)/2.f)/fy;
 float rln=rnorm3df(nx,ny,1.f),T=1.f,D=0.f,M=0.f;int n=0;
 for(uint32_t j=range.x;j<range.y;++j){
  int id=ids[j];float2 d={xy[id].x-x,xy[id].y-y};float4 c=conic[id];
  float power=-0.5f*(c.x*d.x*d.x+c.z*d.y*d.y)-c.y*d.x*d.y;
  if(power>0.f)continue;float a=fminf(0.99f,c.w*expf(power));
  if(a<1.f/255.f)continue;float next=T*(1.f-a);if(next<0.0001f)break;
  float4 pl=plane[id],normal=normals[id];
  float t=pl.x*d.x+pl.y*d.y+pl.z,w=T*a;
  if(n<K){int offset=(r*K+n)*5;
   records[offset]=t*rln;records[offset+1]=w;
   records[offset+2]=normal.x;records[offset+3]=normal.y;records[offset+4]=normal.z;
   member_ids[r*K+n]=id;
  }
  D+=t*w;if(T>0.5f)M=t;T=next;++n;
 }
 counts[r]=n;replay[r*3]=n?D*rln/(1.f-T):0.f;
 replay[r*3+1]=M*rln;replay[r*3+2]=1.f-T;
}
std::vector<torch::Tensor> read_rays(torch::Tensor geom,torch::Tensor bin,
 torch::Tensor img,torch::Tensor pixels,int P,int W,int H,double fx,double fy,int K){
 TORCH_CHECK(geom.is_cuda()&&bin.is_cuda()&&img.is_cuda()&&pixels.is_cuda(),"CUDA tensors required");
 TORCH_CHECK(pixels.scalar_type()==torch::kInt32&&pixels.is_contiguous(),"int32 contiguous pixel indices required");
 TORCH_CHECK(P>0&&W>0&&H>0&&K>0&&pixels.numel()>0,"Invalid dimensions");
 TORCH_CHECK(pixels.min().item<int>()>=0&&pixels.max().item<int>()<W*H,"Pixel out of bounds");
 char* g=(char*)geom.data_ptr();take<float>(g,P);auto pl=take<float4>(g,P);
 auto normals=take<float4>(g,P);take<bool>(g,P*3);take<int>(g,P);
 auto xy=take<float2>(g,P);auto c=take<float4>(g,P);
 char* i=(char*)img.data_ptr();take<uint32_t>(i,W*H*2);auto ranges=take<uint2>(i,W*H);
 char* b=(char*)bin.data_ptr();auto ids=take<uint32_t>(b,0);
 int N=pixels.numel();auto rec=torch::empty({N,K,5},geom.options().dtype(torch::kFloat32));
 auto mids=torch::empty({N,K},geom.options().dtype(torch::kInt32));
 auto counts=torch::empty({N},geom.options().dtype(torch::kInt32));
 auto rep=torch::empty({N,3},geom.options().dtype(torch::kFloat32));
 trace<<<(N+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(ranges,ids,xy,c,pl,normals,
 pixels.data_ptr<int>(),W,H,fx,fy,N,K,rec.data_ptr<float>(),mids.data_ptr<int>(),counts.data_ptr<int>(),rep.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {rec,mids,counts,rep};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("read_rays",&read_rays);}
