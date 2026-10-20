// Research-only buffer reader. Layout and acceptance equations from the host
// cuda_rasterizer/rasterizer_impl.cu and render_forward.cu (Inria LICENSE.md).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <vector>
template<class T> T* take(char*& p, size_t n) {
    auto q=(reinterpret_cast<uintptr_t>(p)+127)&~uintptr_t(127);
    auto out=reinterpret_cast<T*>(q); p=reinterpret_cast<char*>(out+n); return out;
}
__global__ void trace(const uint2* ranges,const uint32_t* ids,const float2* xy,
 const float4* conic,const float4* plane,int W,int H,float fx,float fy,
 int start,int N,int K,float* records,int* counts,float* replay) {
 int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=N)return;
 int p=start+r,x=p%W,y=p/W; uint2 range=ranges[(y/16)*((W+15)/16)+x/16];
 float nx=(x-float(W-1)/2.f)/fx,ny=(y-float(H-1)/2.f)/fy;
 float rln=rnorm3df(nx,ny,1.f),T=1.f,D=0.f,M=0.f; int n=0;
 for(uint32_t j=range.x;j<range.y;++j){
  int id=ids[j];float2 d={xy[id].x-x,xy[id].y-y};float4 c=conic[id];
  float power=-0.5f*(c.x*d.x*d.x+c.z*d.y*d.y)-c.y*d.x*d.y;
  if(power>0.f)continue;float a=fminf(0.99f,c.w*expf(power));
  if(a<1.f/255.f)continue;float next=T*(1.f-a);if(next<0.0001f)break;
  float4 pl=plane[id];float t=pl.x*d.x+pl.y*d.y+pl.z;float w=T*a;
  if(n<K){records[(r*K+n)*2]=t*rln;records[(r*K+n)*2+1]=w;}
  D+=t*w;if(T>0.5f)M=t;T=next;++n;
 }
 counts[r]=n;replay[r*3]=n?D*rln/(1.f-T):0.f;
 replay[r*3+1]=M*rln;replay[r*3+2]=1.f-T;
}
std::vector<torch::Tensor> read_rays(torch::Tensor geom,torch::Tensor bin,
 torch::Tensor img,int P,int W,int H,double fx,double fy,int start,int N,int K){
 TORCH_CHECK(geom.is_cuda()&&bin.is_cuda()&&img.is_cuda(),"CUDA buffers required");
 TORCH_CHECK(start>=0&&N>0&&start+N<=W*H&&K>0,"Invalid batch");
 char* g=(char*)geom.data_ptr();take<float>(g,P);auto pl=take<float4>(g,P);
 take<float4>(g,P);take<bool>(g,P*3);take<int>(g,P);auto xy=take<float2>(g,P);
 auto c=take<float4>(g,P);char* i=(char*)img.data_ptr();take<uint32_t>(i,W*H*2);
 auto ranges=take<uint2>(i,W*H);char* b=(char*)bin.data_ptr();
 // The sorted point_list is the first BinningState field, independent of R.
 auto ids=take<uint32_t>(b,0);
 auto rec=torch::empty({N,K,2},geom.options().dtype(torch::kFloat32));
 auto counts=torch::empty({N},geom.options().dtype(torch::kInt32));
 auto rep=torch::empty({N,3},geom.options().dtype(torch::kFloat32));
 trace<<<(N+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(ranges,ids,xy,c,pl,
  W,H,fx,fy,start,N,K,rec.data_ptr<float>(),counts.data_ptr<int>(),rep.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {rec,counts,rep};
}
double median(std::vector<double> v){std::sort(v.begin(),v.end());int n=v.size();return n%2?v[n/2]:(v[n/2-1]+v[n/2])*0.5;}
struct Item {double d,w,x;int order;};
std::vector<torch::Tensor> aggregate(torch::Tensor rec,torch::Tensor counts,
 torch::Tensor expected,bool valley){
 TORCH_CHECK(!rec.is_cuda()&&rec.is_contiguous(),"Contiguous CPU records required");
 int N=rec.size(0),K=rec.size(1);auto a=rec.data_ptr<float>();auto cnt=counts.data_ptr<int>();
 auto exp=expected.data_ptr<float>();auto out=torch::empty({N,3},rec.options());
 // columns: invalid, B fallback, A/B/C group counts, A/B/C winner contribution share
 auto stats=torch::zeros({N,8},rec.options());auto o=out.data_ptr<float>();auto s=stats.data_ptr<float>();
 at::parallel_for(0,N,256,[&](int64_t lo,int64_t hi){for(int r=lo;r<hi;++r){
  int n=cnt[r];TORCH_CHECK(n<=K,"Record overflow");
  std::vector<Item> v;bool bad=false;double W=0,ww=0;
  for(int j=0;j<n;++j){double d=a[(r*K+j)*2],w=a[(r*K+j)*2+1];
   if(!std::isfinite(d)||d<=0||!std::isfinite(w)||w<=0)bad=true;
   v.push_back({d,w,std::log(d),j});W+=w;ww+=w*w;}
  if(!n||bad){for(int m=0;m<3;++m)o[r*3+m]=exp[r];s[r*8]=bad;continue;}
  std::sort(v.begin(),v.end(),[](Item a,Item b){return a.d==b.d?a.order<b.order:a.d<b.d;});
  std::vector<double> gaps;for(int j=1;j<n;++j)if(v[j].x>v[j-1].x)gaps.push_back(v[j].x-v[j-1].x);
  double tau=0;if(gaps.size()>=4){double m=median(gaps);std::vector<double> dev;for(double g:gaps)dev.push_back(std::abs(g-m));tau=m+3*median(dev);}else s[r*8+1]=1;
  std::vector<double> cuts;
  double range=v.back().x-v[0].x;
  if(valley&&range>0){double sum=0,q25=v[0].x,q75=v.back().x;bool got=false;
   for(auto z:v){sum+=z.w;if(!got&&sum>=W*.25){q25=z.x;got=true;}if(sum>=W*.75){q75=z.x;break;}}
   double bw=2*(q75-q25)/std::cbrt(W*W/ww);
   int B=bw>0?int(std::min(64.,std::max(4.,std::ceil(range/bw)))):16;
   std::vector<double> h(B,0),smooth(B);for(auto z:v){int b=std::min(B-1,int((z.x-v[0].x)/range*B));h[b]+=z.w;}
   for(int b=0;b<B;++b)smooth[b]=(h[std::max(0,b-1)]+2*h[b]+h[std::min(B-1,b+1)])*.25;
   std::vector<std::pair<int,int>> peaks;
   for(int b=0;b<B;){int e=b;while(e+1<B&&smooth[e+1]==smooth[b])++e;
    if((b==0||smooth[b]>smooth[b-1])&&(e==B-1||smooth[e]>smooth[e+1]))peaks.push_back({b,e});b=e+1;}
   for(size_t p=1;p<peaks.size();++p){int left=peaks[p-1].second,right=peaks[p].first;
    if(right-left<2)continue;double mn=smooth[left+1];for(int b=left+1;b<right;++b)mn=std::min(mn,smooth[b]);
    if(mn<=.5*std::min(smooth[left],smooth[right])){
     // Choose the first minimum plateau deterministically if multiple valleys tie.
     int b=left+1;while(smooth[b]!=mn)++b;int e=b;while(e+1<right&&smooth[e+1]==mn)++e;
     cuts.push_back(v[0].x+((b+e)*.5+.5)*range/B);
    }}
  }
  for(int m=0;m<3;++m){double bestW=-1,bestD=0,gw=0,gs=0;int groups=0;size_t ci=0;
   for(int j=0;j<=n;++j){bool split=j==n;
    if(j>0&&j<n){if(m==0||(m==1&&gaps.size()<4))split=v[j].d/v[j-1].d-1>.01;
     else if(m==1)split=v[j].x-v[j-1].x>tau;
     else {while(ci<cuts.size()&&v[j].x>=cuts[ci]){split=true;++ci;}}}
    if(split&&gw>0){double mu=gs/gw;if(gw>bestW||(gw==bestW&&mu<bestD)){bestW=gw;bestD=mu;}++groups;gw=0;gs=0;}
    if(j<n){gw+=v[j].w;gs+=v[j].w*v[j].d;}}
   o[r*3+m]=bestD;s[r*8+2+m]=groups;s[r*8+5+m]=bestW/W;
  }
 }});return {out,stats};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("read_rays",&read_rays);m.def("aggregate",&aggregate);}
