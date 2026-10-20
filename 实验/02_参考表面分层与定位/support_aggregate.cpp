// Original accepted depth/weight pairs are input, never recompute visibility.
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <vector>
struct Item {double d,w;int order;};
std::vector<torch::Tensor> aggregate_support(torch::Tensor records,torch::Tensor counts,
 torch::Tensor expected,torch::Tensor reference,torch::Tensor scales,std::vector<double> lambdas){
 TORCH_CHECK(!records.is_cuda()&&records.is_contiguous(),"CPU contiguous records required");
 int N=records.size(0),K=records.size(1),L=lambdas.size();
 auto rec=records.data_ptr<float>();auto cnt=counts.data_ptr<int>();
 auto E=expected.data_ptr<float>();auto Z=reference.data_ptr<float>();auto S=scales.data_ptr<float>();
 auto out=torch::empty({N,2+2*L},records.options());
 // global: missing reference, bad records, fixed empty, fixed mass/A, fixed span/s, fixed count
 // per lambda: fallback, span/s, mass/A, count, objective, lower, upper, |mean-median|/s, (mean-ref)/s
 auto stats=torch::zeros({N,6+9*L},records.options().dtype(torch::kFloat64));
 auto o=out.data_ptr<float>();auto st=stats.data_ptr<double>();int C=2+2*L,Q=6+9*L;
 for(double lambda:lambdas)TORCH_CHECK(lambda>0&&std::isfinite(lambda),"Invalid lambda");
 at::parallel_for(0,N,256,[&](int64_t lo,int64_t hi){for(int row=lo;row<hi;++row){
  float* y=o+row*C;double* t=st+row*Q;for(int j=0;j<C;++j)y[j]=E[row];
  double z=Z[row],s=S[row];bool hit=std::isfinite(z)&&z>0;
  if(!hit){t[0]=1;continue;}y[0]=z;
  TORCH_CHECK(std::isfinite(s)&&s>0,"Invalid voxel z scale");
  int n=cnt[row];TORCH_CHECK(n<=K&&n>=0,"Truncated input");
  std::vector<Item> v;double A=0;bool bad=false;
  for(int j=0;j<n;++j){double d=rec[(row*K+j)*2],w=rec[(row*K+j)*2+1];
   if(!std::isfinite(d)||d<=0||!std::isfinite(w)||w<=0)bad=true;
   v.push_back({d,w,j});A+=w;}
  if(bad||n==0||A<=0){t[1]=bad;t[2]=1;for(int k=0;k<L;++k)t[6+9*k]=1;continue;}
  std::sort(v.begin(),v.end(),[](Item a,Item b){return a.d==b.d?a.order<b.order:a.d<b.d;});
  double fixedW=0,fixedD=0,fmin=0,fmax=0;int fixedN=0;
  for(auto a:v)if(a.d>=z-4*s&&a.d<=z+4*s){if(!fixedN)fmin=a.d;fmax=a.d;++fixedN;fixedW+=a.w;fixedD+=a.w*a.d;}
  if(fixedN){y[1]=fixedD/fixedW;t[3]=fixedW/A;t[4]=(fmax-fmin)/s;t[5]=fixedN;}else t[2]=1;
  // Collapse identical depths. Every selected endpoint includes the whole equal-depth atom.
  std::vector<double> d,w;std::vector<int> begin,end;
  for(int j=0;j<n;){int e=j+1;double mass=v[j].w;while(e<n&&v[e].d==v[j].d){mass+=v[e].w;++e;}
   d.push_back(v[j].d);w.push_back(mass);begin.push_back(j);end.push_back(e);j=e;}
  int left=std::lower_bound(d.begin(),d.end(),z)-d.begin();
  int right=std::upper_bound(d.begin(),d.end(),z)-d.begin();
  for(int k=0;k<L;++k){double* q=t+6+9*k;double beta=lambdas[k]/s;
   double bestLeft=0,bestRight=0,l=z,u=z,mass=0;int li=left,ri=right-1;
   for(int j=left-1;j>=0;--j){mass+=w[j];double score=mass/A-beta*(z-d[j]);
    if(score>bestLeft||(score==bestLeft&&z-d[j]<z-l)){bestLeft=score;l=d[j];li=j;}}
   mass=0;for(int j=right;j<(int)d.size();++j){mass+=w[j];double score=mass/A-beta*(d[j]-z);
    if(score>bestRight||(score==bestRight&&d[j]-z<u-z)){bestRight=score;u=d[j];ri=j;}}
   // li..ri includes equal-depth-at-reference atoms, if present.
   if(li>ri){q[0]=1;continue;}
   int first=begin[li],last=end[ri];double W=0,D=0;
   for(int j=first;j<last;++j){W+=v[j].w;D+=v[j].w*v[j].d;}
   double score=W/A-beta*(u-l);
   if(!(score>0&&W>0)){q[0]=1;continue;}
   double mean=D/W,median=v[last-1].d,cumulative=0;
   for(int j=first;j<last;++j){cumulative+=v[j].w;if(cumulative>=W*.5){median=v[j].d;break;}}
   y[2+2*k]=mean;y[3+2*k]=median;
   q[1]=(u-l)/s;q[2]=W/A;q[3]=last-first;q[4]=score;q[5]=l;q[6]=u;
   q[7]=std::abs(mean-median)/s;q[8]=(mean-z)/s;
  }
 }});return {out,stats};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("aggregate_support",&aggregate_support);}
