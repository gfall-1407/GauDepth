// Research-only readout derived from the preserved host forward (Inria LICENSE.md).
// Layout checked against rasterizer_impl.cu::fromChunk, NOT struct declaration order.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
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
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("read_rays",&read_rays);m.def("read_pixel_stats",&read_pixel_stats);}
