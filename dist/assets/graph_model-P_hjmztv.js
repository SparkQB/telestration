import{a as Go,c as Se,g as Mo}from"./index-BV2RwRjb.js";function Ko(e,t){for(var n=0;n<t.length;n++){const s=t[n];if(typeof s!="string"&&!Array.isArray(s)){for(const r in s)if(r!=="default"&&!(r in e)){const a=Object.getOwnPropertyDescriptor(s,r);a&&Object.defineProperty(e,r,a.get?a:{enumerable:!0,get:()=>s[r]})}}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ho=1e-7,Xo=1e-4;class j1{constructor(t,n){this.backend=t,this.dataMover=n,this.data=new WeakMap,this.dataIdsCount=0}get(t){return this.data.has(t)||this.dataMover.moveData(this.backend,t),this.data.get(t)}set(t,n){this.dataIdsCount++,this.data.set(t,n)}has(t){return this.data.has(t)}delete(t){return this.dataIdsCount--,this.data.delete(t)}numDataIds(){return this.dataIdsCount}}class Jo{refCount(t){return mt("refCount")}incRef(t){return mt("incRef")}timerAvailable(){return!0}time(t){return mt("time")}read(t){return mt("read")}readSync(t){return mt("readSync")}readToGPU(t,n){return mt("readToGPU")}numDataIds(){return mt("numDataIds")}disposeData(t,n){return mt("disposeData")}write(t,n,s){return mt("write")}move(t,n,s,r,a){return mt("move")}createTensorFromGPUData(t,n,s){return mt("createTensorFromGPUData")}memory(){return mt("memory")}floatPrecision(){return mt("floatPrecision")}epsilon(){return this.floatPrecision()===32?Ho:Xo}dispose(){return mt("dispose")}}function mt(e){throw new Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xr(e){let t=e.length,n=0;for(;t>0;)n=Math.random()*t|0,t--,kn(e,t,n)}function Yo(e,t){if(e.length!==t.length)throw new Error(`Array sizes must match to be shuffled together First array length was ${e.length}Second array length was ${t.length}`);let n=e.length,s=0;for(;n>0;)s=Math.random()*n|0,n--,kn(e,n,s),kn(t,n,s)}function Zo(e,t,n){return Math.max(e,Math.min(t,n))}function Qo(e){return e%2===0?e:e+1}function kn(e,t,n){const s=e[t];e[t]=e[n],e[n]=s}function ti(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function ei(e,t){const n=Math.random();return t*n+(1-n)*e}function ni(e,t){let n=0;for(let s=0;s<e.length;s++){const r=Number(e[s])-Number(t[s]);n+=r*r}return n}function g(e,t){if(!e)throw new Error(typeof t=="string"?t:t())}function lt(e,t,n=""){g(Pt(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function $e(e){g(e!=null,()=>"The input to the tensor constructor must be a non-null value.")}function Z(e){if(e.length===0)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function si(e){return e.length===0}function Jr(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==null&&t[n]!==null&&e[n]!==t[n])return!1;return!0}function Pt(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function Le(e){return e%1===0}function ri(e){if(Math.tanh!=null)return Math.tanh(e);if(e===1/0)return 1;if(e===-1/0)return-1;{const t=Math.exp(2*e);return(t-1)/(t+1)}}function ai(e){const t=Math.ceil(Math.sqrt(e));return[t,Math.ceil(e/t)]}function oi(e){const t=new Uint32Array(e);for(let n=0;n<e;++n)t[n]=n;return Xr(t),t}function Je(e,t){return t<=e.length?e:e+" ".repeat(t-e.length)}function ii(e,t=r=>0,n,s){return new Promise((r,a)=>{let o=0;const i=()=>{if(e()){r();return}o++;const u=t(o);if(n!=null&&o>=n){a();return}s!=null?s(i,u):setTimeout(i,u)};i()})}function ui(e,t){let n=1,s=-1;for(let a=0;a<e.length;++a)if(e[a]>=0)n*=e[a];else if(e[a]===-1){if(s!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${s} and dim ${a}`);s=a}else if(e[a]<0)throw Error(`Shapes can not be < 0. Found ${e[a]} at dim ${a}`);if(s===-1){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(n===0)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);const r=e.slice();return r[s]=t/n,r}function ln(e,t){const n=t.length;return e=e==null?t.map((s,r)=>r):[].concat(e),g(e.every(s=>s>=-n&&s<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),g(e.every(s=>Le(s)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(s=>s<0?n+s:s)}function Yr(e,t){const n=[],s=[],r=t!=null&&Array.isArray(t)&&t.length===0,a=t==null||r?null:ln(t,e).sort();let o=0;for(let i=0;i<e.length;++i){if(a!=null){if(a[o]===i&&e[i]!==1)throw new Error(`Can't squeeze axis ${i} since its dim '${e[i]}' is not 1`);(a[o]==null||a[o]>i)&&e[i]===1&&(n.push(e[i]),s.push(i)),a[o]<=i&&o++}e[i]!==1&&(n.push(e[i]),s.push(i))}return{newShape:n,keptDims:s}}function Zr(e,t){return Cs(e,t)}function Cs(e,t){let n=null;if(e==null||e==="float32")n=new Float32Array(t);else if(e==="int32")n=new Int32Array(t);else if(e==="bool")n=new Uint8Array(t);else if(e==="string")n=new Array(t);else throw new Error(`Unknown data type ${e}`);return n}function Qr(e,t){for(let n=0;n<e.length;n++){const s=e[n];if(isNaN(s)||!isFinite(s))throw Error(`A tensor of type ${t} being uploaded contains ${s}.`)}}function ta(e){return e==="bool"||e==="complex64"||e==="float32"||e==="int32"||e==="string"}function ci(e,t){return!(t==="complex64"||t==="float32"&&e!=="complex64"||t==="int32"&&e!=="float32"&&e!=="complex64"||t==="bool"&&e==="bool")}function En(e){if(e==="float32"||e==="int32")return 4;if(e==="complex64")return 8;if(e==="bool")return 1;throw new Error(`Unknown dtype ${e}`)}function ea(e){if(e==null)return 0;let t=0;return e.forEach(n=>t+=n.length),t}function Ln(e){return typeof e=="string"||e instanceof String}function na(e){return typeof e=="boolean"}function sa(e){return typeof e=="number"}function hn(e){return Array.isArray(e)?hn(e[0]):e instanceof Float32Array?"float32":e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?"int32":sa(e)?"float32":Ln(e)?"string":na(e)?"bool":"float32"}function Qt(e){return!!(e&&e.constructor&&e.call&&e.apply)}function li(e,t){for(let n=t;n<e;++n)if(e%n===0)return n;return e}function pn(e){const t=e.length;if(t<2)return[];const n=new Array(t-1);n[t-2]=e[t-1];for(let s=t-3;s>=0;--s)n[s]=n[s+1]*e[s+1];return n}function ra(e,t,n,s=!1){const r=new Array;if(t.length===1){const a=t[0]*(s?2:1);for(let o=0;o<a;o++)r[o]=n[e+o]}else{const a=t[0],o=t.slice(1),i=o.reduce((u,c)=>u*c)*(s?2:1);for(let u=0;u<a;u++)r[u]=ra(e+u*i,o,n,s)}return r}function me(e,t,n=!1){if(e.length===0)return t[0];const s=e.reduce((r,a)=>r*a)*(n?2:1);if(s===0)return[];if(s!==t.length)throw new Error(`[${e}] does not match the input size ${t.length}${n?" for a complex tensor":""}.`);return ra(0,e,t,n)}function hi(e,t){if(Array.isArray(e))return e;if(t==="float32")return e instanceof Float32Array?e:new Float32Array(e);if(t==="int32")return e instanceof Int32Array?e:new Int32Array(e);if(t==="bool"||t==="string")return Uint8Array.from(new Int32Array(e));throw new Error(`Unknown dtype ${t}`)}function Bs(e,t){const n=zn(e,t);for(let s=0;s<n.length;s++)n[s]=1;return n}function zn(e,t){if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool")return new Uint8Array(e);throw new Error(`Unknown data type ${t}`)}function pi(e,t){const n=e.reduce((s,r)=>s*r,1);if(t==null||t==="float32")return me(e,new Float32Array(n));if(t==="int32")return me(e,new Int32Array(n));if(t==="bool")return me(e,new Uint8Array(n));throw new Error(`Unknown data type ${t}`)}function bt(e){e.forEach(t=>{g(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function fi(e,t,n){if(t===0)return 0;if(t===1)return e[0];let s=e[e.length-1];for(let r=0;r<e.length-1;++r)s+=n[r]*e[r];return s}function mi(e,t,n){if(t===0)return[];if(t===1)return[e];const s=new Array(t);for(let r=0;r<s.length-1;++r)s[r]=Math.floor(e/n[r]),e-=s[r]*n[r];return s[s.length-1]=e,s}function te(e){return e&&e.then&&typeof e.then=="function"}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gr="tfjsflags";class di{constructor(t){this.global=t,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=gi,this.populateURLFlags()}setPlatform(t,n){this.platform!=null&&(C().getBool("IS_TEST")||C().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${t}.`)),this.platformName=t,this.platform=n}registerFlag(t,n,s){if(this.flagRegistry[t]={evaluationFn:n,setHook:s},this.urlFlags[t]!=null){const r=this.urlFlags[t];C().getBool("IS_TEST")||C().getBool("PROD")||console.warn(`Setting feature override from URL ${t}: ${r}.`),this.set(t,r)}}async getAsync(t){return t in this.flags?this.flags[t]:(this.flags[t]=await this.evaluateFlag(t),this.flags[t])}get(t){if(t in this.flags)return this.flags[t];const n=this.evaluateFlag(t);if(te(n))throw new Error(`Flag ${t} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[t]=n,this.flags[t]}getNumber(t){return this.get(t)}getBool(t){return this.get(t)}getString(t){return this.get(t)}getFlags(){return this.flags}get features(){return this.flags}set(t,n){if(this.flagRegistry[t]==null)throw new Error(`Cannot set flag ${t} as it has not been registered.`);this.flags[t]=n,this.flagRegistry[t].setHook!=null&&this.flagRegistry[t].setHook(n)}evaluateFlag(t){if(this.flagRegistry[t]==null)throw new Error(`Cannot evaluate flag '${t}': no evaluation function found.`);return this.flagRegistry[t].evaluationFn()}setFlags(t){this.flags=Object.assign({},t)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const t=this.getQueryParams(this.global.location.search);gr in t&&t[gr].split(",").forEach(s=>{const[r,a]=s.split(":");this.urlFlags[r]=bi(r,a)})}}function gi(e){const t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(n,...s)=>(yi(t,s[0],s[1]),s.join("="))),t}function yi(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||"")}function bi(e,t){const n=t.toLowerCase();return n==="true"||n==="false"?n==="true":`${+n}`===n?+n:t}function C(){return aa}let aa=null;function wi(e){aa=e}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Jn;function oa(){if(Jn==null){let e;if(typeof window<"u")e=window;else if(typeof global<"u")e=global;else if(typeof process<"u")e=process;else if(typeof self<"u")e=self;else throw new Error("Could not find a global object");Jn=e}return Jn}function Ni(){const e=oa();return e._tfGlobals==null&&(e._tfGlobals=new Map),e._tfGlobals}function Rs(e,t){const n=Ni();if(n.has(e))return n.get(e);{const s=t();return n.set(e,s),n.get(e)}}const Ti="Abs",Si="Acos",$i="Acosh",ia="Add",ki="AddN",Ei="All",vi="Any",_i="ArgMax",xi="ArgMin",Ii="Asin",Ai="Asinh",Di="Atan",Oi="Atanh",Fi="Atan2",Ci="AvgPool",W1="AvgPoolGrad",Bi="AvgPool3D",q1="AvgPool3DGrad",Ri="BatchMatMul",Pi="BatchToSpaceND",Li="Bincount",zi="BitwiseAnd",G1="BroadcastTo",Vi="BroadcastArgs",ua="Cast",Ui="Ceil",ji="ClipByValue",Wi="Complex",qi="ComplexAbs",Gi="Concat",Mi="Conv2D",Ki="Conv2DBackpropFilter",Hi="Conv2DBackpropInput",Xi="Conv3D",M1="Conv3DBackpropFilterV2",Ji="Conv3DBackpropInputV2",Yi="Cos",Zi="Cosh",Qi="Cumprod",tu="Cumsum",eu="CropAndResize",nu="DenseBincount",su="DepthToSpace",ru="DepthwiseConv2dNative",au="DepthwiseConv2dNativeBackpropFilter",ou="DepthwiseConv2dNativeBackpropInput",iu="Diag",uu="Dilation2D",K1="Dilation2DBackpropInput",H1="Dilation2DBackpropFilter",ca="Draw",cu="RealDiv",lu="Einsum",hu="Elu",X1="EluGrad",pu="Erf",fu="Equal",mu="Exp",du="ExpandDims",gu="Expm1",yu="FFT",bu="Fill",wu="FlipLeftRight",Nu="Floor",Tu="FloorDiv",Su="FusedBatchNorm",$u="GatherV2",ku="GatherNd",Eu="Greater",vu="GreaterEqual",la="Identity",_u="IFFT",xu="Imag",Iu="IsFinite",Au="IsInf",Du="IsNan",Ou="LeakyRelu",Fu="Less",Cu="LessEqual",Bu="LinSpace",Ru="Log",Pu="Log1p",Lu="LogicalAnd",zu="LogicalNot",Vu="LogicalOr",J1="LogicalXor",Y1="LogSoftmax",Z1="LowerBound",Uu="LRN",Q1="LRNGrad",tN="MatrixBandPart",ju="Max",Wu="Maximum",qu="MaxPool",eN="MaxPoolGrad",Gu="MaxPool3D",nN="MaxPool3DGrad",Mu="MaxPoolWithArgmax",Ku="Mean",Hu="Min",Xu="Minimum",Ju="MirrorPad",Yu="Mod",Zu="Multinomial",Qu="Multiply",tc="Neg",ec="NotEqual",nc="NonMaxSuppressionV3",sc="NonMaxSuppressionV4",rc="NonMaxSuppressionV5",ac="OnesLike",oc="OneHot",ic="Pack",uc="PadV2",sN="Pool",cc="Pow",lc="Prelu",hc="Prod",pc="RaggedGather",fc="RaggedRange",mc="RaggedTensorToTensor",dc="Range",gc="Real",yc="Reciprocal",bc="Relu",wc="Reshape",Nc="ResizeNearestNeighbor",rN="ResizeNearestNeighborGrad",Tc="ResizeBilinear",aN="ResizeBilinearGrad",Sc="Relu6",$c="Reverse",kc="Round",Ec="Rsqrt",vc="ScatterNd",_c="TensorScatterUpdate",xc="SearchSorted",Ic="Select",Ac="Selu",Dc="Slice",Oc="Sin",Fc="Sinh",Cc="Sign",Bc="Sigmoid",Rc="Softplus",Pc="Sqrt",Lc="Sum",zc="SpaceToBatchND",Vc="SplitV",Uc="Softmax",jc="SparseFillEmptyRows",Wc="SparseReshape",qc="SparseSegmentMean",Gc="SparseSegmentSum",Mc="SparseToDense",Kc="SquaredDifference",oN="Square",Hc="StaticRegexReplace",Xc="StridedSlice",Jc="StringNGrams",Yc="StringSplit",Zc="StringToHashBucketFast",Qc="Sub",tl="Tan",el="Tanh",ha="Tile",nl="TopK",sl="Transform",Yn="Transpose",rl="Unique",al="Unpack",ol="UnsortedSegmentSum",iN="UpperBound",il="ZerosLike",ul="Step",yr="FromPixels",cl="RotateWithOffset",br="_FusedMatMul",wr="FusedConv2D",Nr="FusedDepthwiseConv2D";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ue(...e){C().getBool("IS_TEST")||C().getBool("PROD")||console.warn(...e)}function uN(...e){C().getBool("IS_TEST")||C().getBool("PROD")||console.log(...e)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ze=Rs("kernelRegistry",()=>new Map),Qe=Rs("gradRegistry",()=>new Map);function vn(e,t){const n=Ps(e,t);return ze.get(n)}function Tr(e){return Qe.get(e)}function ss(e){const t=ze.entries(),n=[];for(;;){const{done:s,value:r}=t.next();if(s)break;const[a,o]=r,[i]=a.split("_");i===e&&n.push(o)}return n}function ll(e){const{kernelName:t,backendName:n}=e,s=Ps(t,n);ze.has(s)&&ue(`The kernel '${t}' for backend '${n}' is already registered`),ze.set(s,e)}function cN(e){const{kernelName:t}=e;Qe.has(t)&&C().getBool("DEBUG")&&ue(`Overriding the gradient for '${t}'`),Qe.set(t,e)}function lN(e,t){const n=Ps(e,t);if(!ze.has(n))throw new Error(`The kernel '${e}' for backend '${t}' is not registered`);ze.delete(n)}function hN(e){if(!Qe.has(e))throw new Error(`The gradient '${e}' for backend is not registered`);Qe.delete(e)}function pN(e,t){ss(e).forEach(s=>{const r=Object.assign({},s,{backendName:t});ll(r)})}function Ps(e,t){return`${t}_${e}`}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pa(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}var fa=H,Et=null;try{Et=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function H(e,t,n){this.low=e|0,this.high=t|0,this.unsigned=!!n}H.prototype.__isLong__;Object.defineProperty(H.prototype,"__isLong__",{value:!0});function wt(e){return(e&&e.__isLong__)===!0}H.isLong=wt;var Sr={},$r={};function ke(e,t){var n,s,r;return t?(e>>>=0,(r=0<=e&&e<256)&&(s=$r[e],s)?s:(n=X(e,(e|0)<0?-1:0,!0),r&&($r[e]=n),n)):(e|=0,(r=-128<=e&&e<128)&&(s=Sr[e],s)?s:(n=X(e,e<0?-1:0,!1),r&&(Sr[e]=n),n))}H.fromInt=ke;function vt(e,t){if(isNaN(e))return t?he:_t;if(t){if(e<0)return he;if(e>=ma)return ya}else{if(e<=-Er)return yt;if(e+1>=Er)return ga}return e<0?vt(-e,t).neg():X(e%Ve|0,e/Ve|0,t)}H.fromNumber=vt;function X(e,t,n){return new H(e,t,n)}H.fromBits=X;var _n=Math.pow;function Ls(e,t,n){if(e.length===0)throw Error("empty string");if(e==="NaN"||e==="Infinity"||e==="+Infinity"||e==="-Infinity")return _t;if(typeof t=="number"?(n=t,t=!1):t=!!t,n=n||10,n<2||36<n)throw RangeError("radix");var s;if((s=e.indexOf("-"))>0)throw Error("interior hyphen");if(s===0)return Ls(e.substring(1),t,n).neg();for(var r=vt(_n(n,8)),a=_t,o=0;o<e.length;o+=8){var i=Math.min(8,e.length-o),u=parseInt(e.substring(o,o+i),n);if(i<8){var c=vt(_n(n,i));a=a.mul(c).add(vt(u))}else a=a.mul(r),a=a.add(vt(u))}return a.unsigned=t,a}H.fromString=Ls;function Dt(e,t){return typeof e=="number"?vt(e,t):typeof e=="string"?Ls(e,t):X(e.low,e.high,typeof t=="boolean"?t:e.unsigned)}H.fromValue=Dt;var kr=65536,hl=1<<24,Ve=kr*kr,ma=Ve*Ve,Er=ma/2,vr=ke(hl),_t=ke(0);H.ZERO=_t;var he=ke(0,!0);H.UZERO=he;var Fe=ke(1);H.ONE=Fe;var da=ke(1,!0);H.UONE=da;var rs=ke(-1);H.NEG_ONE=rs;var ga=X(-1,2147483647,!1);H.MAX_VALUE=ga;var ya=X(-1,-1,!0);H.MAX_UNSIGNED_VALUE=ya;var yt=X(0,-2147483648,!1);H.MIN_VALUE=yt;var E=H.prototype;E.toInt=function(){return this.unsigned?this.low>>>0:this.low};E.toNumber=function(){return this.unsigned?(this.high>>>0)*Ve+(this.low>>>0):this.high*Ve+(this.low>>>0)};E.toString=function(t){if(t=t||10,t<2||36<t)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative())if(this.eq(yt)){var n=vt(t),s=this.div(n),r=s.mul(n).sub(this);return s.toString(t)+r.toInt().toString(t)}else return"-"+this.neg().toString(t);for(var a=vt(_n(t,6),this.unsigned),o=this,i="";;){var u=o.div(a),c=o.sub(u.mul(a)).toInt()>>>0,h=c.toString(t);if(o=u,o.isZero())return h+i;for(;h.length<6;)h="0"+h;i=""+h+i}};E.getHighBits=function(){return this.high};E.getHighBitsUnsigned=function(){return this.high>>>0};E.getLowBits=function(){return this.low};E.getLowBitsUnsigned=function(){return this.low>>>0};E.getNumBitsAbs=function(){if(this.isNegative())return this.eq(yt)?64:this.neg().getNumBitsAbs();for(var t=this.high!=0?this.high:this.low,n=31;n>0&&!(t&1<<n);n--);return this.high!=0?n+33:n+1};E.isZero=function(){return this.high===0&&this.low===0};E.eqz=E.isZero;E.isNegative=function(){return!this.unsigned&&this.high<0};E.isPositive=function(){return this.unsigned||this.high>=0};E.isOdd=function(){return(this.low&1)===1};E.isEven=function(){return(this.low&1)===0};E.equals=function(t){return wt(t)||(t=Dt(t)),this.unsigned!==t.unsigned&&this.high>>>31===1&&t.high>>>31===1?!1:this.high===t.high&&this.low===t.low};E.eq=E.equals;E.notEquals=function(t){return!this.eq(t)};E.neq=E.notEquals;E.ne=E.notEquals;E.lessThan=function(t){return this.comp(t)<0};E.lt=E.lessThan;E.lessThanOrEqual=function(t){return this.comp(t)<=0};E.lte=E.lessThanOrEqual;E.le=E.lessThanOrEqual;E.greaterThan=function(t){return this.comp(t)>0};E.gt=E.greaterThan;E.greaterThanOrEqual=function(t){return this.comp(t)>=0};E.gte=E.greaterThanOrEqual;E.ge=E.greaterThanOrEqual;E.compare=function(t){if(wt(t)||(t=Dt(t)),this.eq(t))return 0;var n=this.isNegative(),s=t.isNegative();return n&&!s?-1:!n&&s?1:this.unsigned?t.high>>>0>this.high>>>0||t.high===this.high&&t.low>>>0>this.low>>>0?-1:1:this.sub(t).isNegative()?-1:1};E.comp=E.compare;E.negate=function(){return!this.unsigned&&this.eq(yt)?yt:this.not().add(Fe)};E.neg=E.negate;E.add=function(t){wt(t)||(t=Dt(t));var n=this.high>>>16,s=this.high&65535,r=this.low>>>16,a=this.low&65535,o=t.high>>>16,i=t.high&65535,u=t.low>>>16,c=t.low&65535,h=0,p=0,f=0,d=0;return d+=a+c,f+=d>>>16,d&=65535,f+=r+u,p+=f>>>16,f&=65535,p+=s+i,h+=p>>>16,p&=65535,h+=n+o,h&=65535,X(f<<16|d,h<<16|p,this.unsigned)};E.subtract=function(t){return wt(t)||(t=Dt(t)),this.add(t.neg())};E.sub=E.subtract;E.multiply=function(t){if(this.isZero())return _t;if(wt(t)||(t=Dt(t)),Et){var n=Et.mul(this.low,this.high,t.low,t.high);return X(n,Et.get_high(),this.unsigned)}if(t.isZero())return _t;if(this.eq(yt))return t.isOdd()?yt:_t;if(t.eq(yt))return this.isOdd()?yt:_t;if(this.isNegative())return t.isNegative()?this.neg().mul(t.neg()):this.neg().mul(t).neg();if(t.isNegative())return this.mul(t.neg()).neg();if(this.lt(vr)&&t.lt(vr))return vt(this.toNumber()*t.toNumber(),this.unsigned);var s=this.high>>>16,r=this.high&65535,a=this.low>>>16,o=this.low&65535,i=t.high>>>16,u=t.high&65535,c=t.low>>>16,h=t.low&65535,p=0,f=0,d=0,y=0;return y+=o*h,d+=y>>>16,y&=65535,d+=a*h,f+=d>>>16,d&=65535,d+=o*c,f+=d>>>16,d&=65535,f+=r*h,p+=f>>>16,f&=65535,f+=a*c,p+=f>>>16,f&=65535,f+=o*u,p+=f>>>16,f&=65535,p+=s*h+r*c+a*u+o*i,p&=65535,X(d<<16|y,p<<16|f,this.unsigned)};E.mul=E.multiply;E.divide=function(t){if(wt(t)||(t=Dt(t)),t.isZero())throw Error("division by zero");if(Et){if(!this.unsigned&&this.high===-2147483648&&t.low===-1&&t.high===-1)return this;var n=(this.unsigned?Et.div_u:Et.div_s)(this.low,this.high,t.low,t.high);return X(n,Et.get_high(),this.unsigned)}if(this.isZero())return this.unsigned?he:_t;var s,r,a;if(this.unsigned){if(t.unsigned||(t=t.toUnsigned()),t.gt(this))return he;if(t.gt(this.shru(1)))return da;a=he}else{if(this.eq(yt)){if(t.eq(Fe)||t.eq(rs))return yt;if(t.eq(yt))return Fe;var o=this.shr(1);return s=o.div(t).shl(1),s.eq(_t)?t.isNegative()?Fe:rs:(r=this.sub(t.mul(s)),a=s.add(r.div(t)),a)}else if(t.eq(yt))return this.unsigned?he:_t;if(this.isNegative())return t.isNegative()?this.neg().div(t.neg()):this.neg().div(t).neg();if(t.isNegative())return this.div(t.neg()).neg();a=_t}for(r=this;r.gte(t);){s=Math.max(1,Math.floor(r.toNumber()/t.toNumber()));for(var i=Math.ceil(Math.log(s)/Math.LN2),u=i<=48?1:_n(2,i-48),c=vt(s),h=c.mul(t);h.isNegative()||h.gt(r);)s-=u,c=vt(s,this.unsigned),h=c.mul(t);c.isZero()&&(c=Fe),a=a.add(c),r=r.sub(h)}return a};E.div=E.divide;E.modulo=function(t){if(wt(t)||(t=Dt(t)),Et){var n=(this.unsigned?Et.rem_u:Et.rem_s)(this.low,this.high,t.low,t.high);return X(n,Et.get_high(),this.unsigned)}return this.sub(this.div(t).mul(t))};E.mod=E.modulo;E.rem=E.modulo;E.not=function(){return X(~this.low,~this.high,this.unsigned)};E.and=function(t){return wt(t)||(t=Dt(t)),X(this.low&t.low,this.high&t.high,this.unsigned)};E.or=function(t){return wt(t)||(t=Dt(t)),X(this.low|t.low,this.high|t.high,this.unsigned)};E.xor=function(t){return wt(t)||(t=Dt(t)),X(this.low^t.low,this.high^t.high,this.unsigned)};E.shiftLeft=function(t){return wt(t)&&(t=t.toInt()),(t&=63)===0?this:t<32?X(this.low<<t,this.high<<t|this.low>>>32-t,this.unsigned):X(0,this.low<<t-32,this.unsigned)};E.shl=E.shiftLeft;E.shiftRight=function(t){return wt(t)&&(t=t.toInt()),(t&=63)===0?this:t<32?X(this.low>>>t|this.high<<32-t,this.high>>t,this.unsigned):X(this.high>>t-32,this.high>=0?0:-1,this.unsigned)};E.shr=E.shiftRight;E.shiftRightUnsigned=function(t){if(wt(t)&&(t=t.toInt()),t&=63,t===0)return this;var n=this.high;if(t<32){var s=this.low;return X(s>>>t|n<<32-t,n>>>t,this.unsigned)}else return t===32?X(n,0,this.unsigned):X(n>>>t-32,0,this.unsigned)};E.shru=E.shiftRightUnsigned;E.shr_u=E.shiftRightUnsigned;E.toSigned=function(){return this.unsigned?X(this.low,this.high,!1):this};E.toUnsigned=function(){return this.unsigned?this:X(this.low,this.high,!0)};E.toBytes=function(t){return t?this.toBytesLE():this.toBytesBE()};E.toBytesLE=function(){var t=this.high,n=this.low;return[n&255,n>>>8&255,n>>>16&255,n>>>24,t&255,t>>>8&255,t>>>16&255,t>>>24]};E.toBytesBE=function(){var t=this.high,n=this.low;return[t>>>24,t>>>16&255,t>>>8&255,t&255,n>>>24,n>>>16&255,n>>>8&255,n&255]};H.fromBytes=function(t,n,s){return s?H.fromBytesLE(t,n):H.fromBytesBE(t,n)};H.fromBytesLE=function(t,n){return new H(t[0]|t[1]<<8|t[2]<<16|t[3]<<24,t[4]|t[5]<<8|t[6]<<16|t[7]<<24,n)};H.fromBytesBE=function(t,n){return new H(t[4]<<24|t[5]<<16|t[6]<<8|t[7],t[0]<<24|t[1]<<16|t[2]<<8|t[3],n)};const ba=Go(fa),pl=Ko({__proto__:null,default:ba},[fa]);/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ce=ba||pl;function fn(e){return ce.fromString(e,!0,16)}const wa=fn("c3a5c85c97cb3127"),ie=fn("b492b66fbe98f273"),ut=fn("9ae16a3b2f90404f");function as(e){return e.xor(e.shru(47))}function Na(e,t,n){const s=e.slice(t,t+n);return ce.fromBytes(Array.from(s),!0,!0)}function W(e,t){return Na(e,t,8)}function _r(e,t){return Na(e,t,4)}function rt(e,t){return t===0?e:e.shru(t).or(e.shl(64-t))}function Jt(e,t,n=fn("9ddfea08eb382d69")){let s=e.xor(t).mul(n);s=s.xor(s.shru(47));let r=t.xor(s).mul(n);return r=r.xor(r.shru(47)),r=r.mul(n),r}function fl(e,t,n,s,r,a){r=r.add(e),a=rt(a.add(r).add(s),21);const o=r;return r=r.add(t),r=r.add(n),a=a.add(rt(r,44)),[r.add(s),a.add(o)]}function yn(e,t,n,s){return fl(W(e,t),W(e,t+8),W(e,t+16),W(e,t+24),n,s)}function ml(e,t=e.length){if(t>=8){const n=ut.add(t*2),s=W(e,0).add(ut),r=W(e,t-8),a=rt(r,37).mul(n).add(s),o=rt(s,25).add(r).mul(n);return Jt(a,o,n)}if(t>=4){const n=ut.add(t*2),s=_r(e,0);return Jt(s.shl(3).add(t),_r(e,t-4),n)}if(t>0){const n=e[0],s=e[t>>1],r=e[t-1],a=n+(s<<8),o=t+(r<<2);return as(ut.mul(a).xor(wa.mul(o))).mul(ut)}return ut}function dl(e,t=e.length){const n=ut.add(t*2),s=W(e,0).mul(ie),r=W(e,8),a=W(e,t-8).mul(n),o=W(e,t-16).mul(ut);return Jt(rt(s.add(r),43).add(rt(a,30)).add(o),s.add(rt(r.add(ut),18)).add(a),n)}function gl(e,t=e.length){const n=ut.add(t*2),s=W(e,0).mul(ut),r=W(e,8),a=W(e,t-8).mul(n),o=W(e,t-16).mul(ut),i=rt(s.add(r),43).add(rt(a,30)).add(o),u=Jt(i,s.add(rt(r.add(ut),18)).add(a),n),c=W(e,16).mul(n),h=W(e,24),p=i.add(W(e,t-32)).mul(n),f=u.add(W(e,t-24)).mul(n);return Jt(rt(c.add(h),43).add(rt(p,30)).add(f),c.add(rt(h.add(s),18)).add(p),n)}function yl(e,t=e.length){const n=ce.fromNumber(81,!0);if(t<=32)return t<=16?ml(e,t):dl(e,t);if(t<=64)return gl(e,t);let s=n,r=n.mul(ie).add(113),a=as(r.mul(ut).add(113)).mul(ut),o=[ce.UZERO,ce.UZERO],i=[ce.UZERO,ce.UZERO];s=s.mul(ut).add(W(e,0));let u=0;const c=(t-1>>6)*64,h=c+(t-1&63)-63;do s=rt(s.add(r).add(o[0]).add(W(e,u+8)),37).mul(ie),r=rt(r.add(o[1]).add(W(e,u+48)),42).mul(ie),s=s.xor(i[1]),r=r.add(o[0]).add(W(e,u+40)),a=rt(a.add(i[0]),33).mul(ie),o=yn(e,u,o[1].mul(ie),s.add(i[0])),i=yn(e,u+32,a.add(i[1]),r.add(W(e,u+16))),[a,s]=[s,a],u+=64;while(u!==c);const p=ie.add(a.and(255).shl(1));return u=h,i[0]=i[0].add(t-1&63),o[0]=o[0].add(i[0]),i[0]=i[0].add(o[0]),s=rt(s.add(r).add(o[0]).add(W(e,u+8)),37).mul(p),r=rt(r.add(o[1]).add(W(e,u+48)),42).mul(p),s=s.xor(i[1].mul(9)),r=r.add(o[0].mul(9).add(W(e,u+40))),a=rt(a.add(i[0]),33).mul(p),o=yn(e,u,o[1].mul(p),s.add(i[0])),i=yn(e,u+32,a.add(i[1]),r.add(W(e,u+16))),[a,s]=[s,a],Jt(Jt(o[0],i[0],p).add(as(r).mul(wa)).add(a),Jt(o[1],i[1],p).add(s),p)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bl(e,t){return t==="string"?zs(e):Vn([e],t)}function wl(e,t){return e instanceof Float32Array&&t==="float32"||e instanceof Int32Array&&t==="int32"||e instanceof Uint8Array&&t==="bool"}function Vn(e,t){if(t==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(e)&&(e=Ue(e)),C().getBool("DEBUG")&&Qr(e,t),wl(e,t))return e;if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool"){const n=new Uint8Array(e.length);for(let s=0;s<n.length;++s)Math.round(e[s])!==0&&(n[s]=1);return n}else throw new Error(`Unknown data type ${t}`)}function tn(){return C().platform.now()}function Nl(e,t){return C().platform.fetch(e,t)}function zs(e,t="utf-8"){return t=t||"utf-8",C().platform.encode(e,t)}function os(e,t="utf-8"){return t=t||"utf-8",C().platform.decode(e,t)}function St(e){return C().platform.isTypedArray!=null?C().platform.isTypedArray(e):pa(e)}function Ue(e,t=[],n=!1){if(t==null&&(t=[]),typeof e=="boolean"||typeof e=="number"||typeof e=="string"||te(e)||e==null||St(e)&&n)t.push(e);else if(Array.isArray(e)||St(e))for(let s=0;s<e.length;++s)Ue(e[s],t,n);else{let s=-1;for(const r of Object.keys(e))/^([1-9]+[0-9]*|0)$/.test(r)&&(s=Math.max(s,Number(r)));for(let r=0;r<=s;r++)Ue(e[r],t,n)}return t}const fN=Object.freeze(Object.defineProperty({__proto__:null,arraysEqual:Pt,arraysEqualWithNull:Jr,assert:g,assertNonNegativeIntegerDimensions:bt,assertNonNull:$e,assertShapesMatch:lt,bytesFromStringArray:ea,bytesPerElement:En,checkConversionForErrors:Qr,clamp:Zo,computeStrides:pn,convertBackendValuesAndArrayBuffer:hi,createScalarValue:bl,createShuffledIndices:oi,decodeString:os,distSquared:ni,encodeString:zs,fetch:Nl,fingerPrint64:yl,flatten:Ue,getArrayFromDType:Cs,getTypedArrayFromDType:Zr,hasEncodingLoss:ci,hexToLong:fn,indexToLoc:mi,inferDtype:hn,inferFromImplicitShape:ui,isBoolean:na,isFunction:Qt,isInt:Le,isNumber:sa,isPromise:te,isScalarShape:si,isString:Ln,isTypedArray:St,isValidDtype:ta,locToIndex:fi,makeOnesTypedArray:Bs,makeZerosNestedTypedArray:pi,makeZerosTypedArray:zn,nearestDivisor:li,nearestLargerEven:Qo,now:tn,parseAxisParam:ln,randUniform:ei,repeatedTry:ii,rightPad:Je,shuffle:Xr,shuffleCombo:Yo,sizeFromShape:Z,sizeToSquarishShape:ai,squeezeShape:Yr,sum:ti,swap:kn,tanh:ri,toNestedArray:me,toTypedArray:Vn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Tl{constructor(t,n){this.backendTimer=t,this.logger=n,n==null&&(this.logger=new $l)}profileKernel(t,n,s){let r;const a=()=>{r=s()};let o;const i=tn();if(this.backendTimer.timerAvailable())o=this.backendTimer.time(a);else{a();for(const c of r)c.dataSync();o=Promise.resolve({kernelMs:tn()-i})}if(C().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let c=0;c<r.length;c++){const h=r[c];h.data().then(p=>{Sl(p,h.dtype,t)})}return{kernelName:t,outputs:r,inputs:n,timeMs:o.then(c=>c.kernelMs),extraInfo:o.then(c=>c.getExtraProfileInfo!=null?c.getExtraProfileInfo():"")}}logKernelProfile(t){const{kernelName:n,outputs:s,timeMs:r,inputs:a,extraInfo:o}=t;s.forEach(i=>{Promise.all([i.data(),r,o]).then(u=>{this.logger.logKernelProfile(n,i,u[0],u[1],a,u[2])})})}}function Sl(e,t,n){if(t!=="float32")return!1;for(let s=0;s<e.length;s++){const r=e[s];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${n}'`),!0}return!1}class $l{logKernelProfile(t,n,s,r,a,o){const i=typeof r=="number"?Je(`${r}ms`,9):r.error,u=Je(t,25),c=n.rank,h=n.size,p=Je(n.shape.toString(),14);let f="";for(const d in a){const y=a[d];if(y!=null){const T=y.shape||n.shape,N=T.length;f+=`${d}: ${N}D ${N>0?T:""} `}}console.log(`%c${u}	%c${i}	%c${c}D ${p}	%c${h}	%c${f}	%c${o}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kl(e,t,n){const s={},r={};for(let u=0;u<t.length;u++)s[t[u].id]=!0;for(let u=0;u<e.length;u++){const c=e[u],h=c.inputs;for(const p in h){const f=h[p];let d=!1;for(let y=0;y<t.length;y++)if(s[f.id]){c.outputs.forEach(T=>s[T.id]=!0),d=!0,r[c.id]=!0;break}if(d)break}}const a={};a[n.id]=!0;const o={};for(let u=e.length-1;u>=0;u--){const c=e[u],h=c.inputs;for(let p=0;p<c.outputs.length;p++)if(a[c.outputs[p].id]){for(const f in h)a[h[f].id]=!0,o[c.id]=!0;break}}const i=[];for(let u=0;u<e.length;u++){const c=e[u];if(r[c.id]&&o[c.id]){const h={};for(const f in c.inputs){const d=c.inputs[f];s[d.id]&&(h[f]=d)}const p=Object.assign({},c);p.inputs=h,p.outputs=c.outputs,i.push(p)}}return i}function El(e,t,n,s){for(let r=t.length-1;r>=0;r--){const a=t[r],o=[];if(a.outputs.forEach(u=>{const c=e[u.id];c!=null?o.push(c):o.push(null)}),a.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${a.kernelName}.`);const i=a.gradient(o);for(const u in a.inputs){if(!(u in i))throw new Error(`Cannot backprop through input ${u}. Available gradients found: ${Object.keys(i)}.`);const c=n(()=>i[u]());if(c.dtype!=="float32")throw new Error(`Error in gradient for op ${a.kernelName}. The gradient of input ${u} must have 'float32' dtype, but has '${c.dtype}'`);const h=a.inputs[u];if(!Pt(c.shape,h.shape))throw new Error(`Error in gradient for op ${a.kernelName}. The gradient of input '${u}' has shape '${c.shape}', which does not match the shape of the input '${h.shape}'`);if(e[h.id]==null)e[h.id]=c;else{const p=e[h.id];e[h.id]=s(p,c),p.dispose()}}}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xr=20,Me=3,Zn=7;function vl(e,t,n,s){const r=pn(t),a=_l(e,t,n,r),o=t.length,i=Nn(e,t,n,r,a),u=["Tensor"];return s&&(u.push(`  dtype: ${n}`),u.push(`  rank: ${o}`),u.push(`  shape: [${t}]`),u.push("  values:")),u.push(i.map(c=>"    "+c).join(`
`)),u.join(`
`)}function _l(e,t,n,s){const r=Z(t),a=s[s.length-1],o=new Array(a).fill(0),i=t.length,u=n==="complex64"?Xe(e):e;if(i>1)for(let c=0;c<r/a;c++){const h=c*a;for(let p=0;p<a;p++)o[p]=Math.max(o[p],He(u[h+p],0,n).length)}return o}function He(e,t,n){let s;return Array.isArray(e)?s=`${parseFloat(e[0].toFixed(Zn))} + ${parseFloat(e[1].toFixed(Zn))}j`:Ln(e)?s=`'${e}'`:n==="bool"?s=Ta(e):s=parseFloat(e.toFixed(Zn)).toString(),Je(s,t)}function Ta(e){return e===0?"false":"true"}function Nn(e,t,n,s,r,a=!0){const o=n==="complex64"?2:1,i=t[0],u=t.length;if(u===0){if(n==="complex64"){const T=Xe(e);return[He(T[0],0,n)]}return n==="bool"?[Ta(e[0])]:[e[0].toString()]}if(u===1){if(i>xr){const N=Me*o;let S=Array.from(e.slice(0,N)),I=Array.from(e.slice((i-Me)*o,i*o));return n==="complex64"&&(S=Xe(S),I=Xe(I)),["["+S.map((A,k)=>He(A,r[k],n)).join(", ")+", ..., "+I.map((A,k)=>He(A,r[i-Me+k],n)).join(", ")+"]"]}return["["+(n==="complex64"?Xe(e):Array.from(e)).map((N,S)=>He(N,r[S],n)).join(", ")+"]"]}const c=t.slice(1),h=s.slice(1),p=s[0]*o,f=[];if(i>xr){for(let T=0;T<Me;T++){const N=T*p,S=N+p;f.push(...Nn(e.slice(N,S),c,n,h,r,!1))}f.push("...");for(let T=i-Me;T<i;T++){const N=T*p,S=N+p;f.push(...Nn(e.slice(N,S),c,n,h,r,T===i-1))}}else for(let T=0;T<i;T++){const N=T*p,S=N+p;f.push(...Nn(e.slice(N,S),c,n,h,r,T===i-1))}const d=u===2?",":"";f[0]="["+(i>0?f[0]+d:"");for(let T=1;T<f.length-1;T++)f[T]=" "+f[T]+d;let y=`,
`;for(let T=2;T<u;T++)y+=`
`;return f[f.length-1]=" "+f[f.length-1]+"]"+(a?"":y),f}function Xe(e){const t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class is{constructor(t,n,s){if(this.dtype=n,this.shape=t.slice(),this.size=Z(t),s!=null){const r=s.length;g(r===this.size,()=>`Length of values '${r}' does not match the size inferred by the shape '${this.size}'.`)}if(n==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=s||Cs(n,this.size),this.strides=pn(t)}set(t,...n){n.length===0&&(n=[0]),g(n.length===this.rank,()=>`The number of provided coordinates (${n.length}) must match the rank (${this.rank})`);const s=this.locToIndex(n);this.values[s]=t}get(...t){t.length===0&&(t=[0]);let n=0;for(const r of t){if(r<0||r>=this.shape[n]){const a=`Requested out of range element at ${t}.   Buffer shape=${this.shape}`;throw new Error(a)}n++}let s=t[t.length-1];for(let r=0;r<t.length-1;++r)s+=this.strides[r]*t[r];return this.values[s]}locToIndex(t){if(this.rank===0)return 0;if(this.rank===1)return t[0];let n=t[t.length-1];for(let s=0;s<t.length-1;++s)n+=this.strides[s]*t[s];return n}indexToLoc(t){if(this.rank===0)return[];if(this.rank===1)return[t];const n=new Array(this.shape.length);for(let s=0;s<n.length-1;++s)n[s]=Math.floor(t/this.strides[s]),t-=n[s]*this.strides[s];return n[n.length-1]=t,n}get rank(){return this.shape.length}toTensor(){return It().makeTensor(this.values,this.shape,this.dtype)}}let It=null,Ae=null;function xl(e){It=e}function Il(e){Ae=e}class et{constructor(t,n,s,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=t.slice(),this.dtype=n||"float32",this.size=Z(t),this.strides=pn(t),this.dataId=s,this.id=r,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const t=await this.data();return Ae.buffer(this.shape,this.dtype,t)}bufferSync(){return Ae.buffer(this.shape,this.dtype,this.dataSync())}async array(){const t=await this.data();return me(this.shape,t,this.dtype==="complex64")}arraySync(){return me(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const t=It().read(this.dataId);if(this.dtype==="string"){const n=await t;try{return n.map(s=>os(s))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return t}dataToGPU(t){return this.throwIfDisposed(),It().readToGPU(this.dataId,t)}dataSync(){this.throwIfDisposed();const t=It().readSync(this.dataId);if(this.dtype==="string")try{return t.map(n=>os(n))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return t}async bytes(){this.throwIfDisposed();const t=await It().read(this.dataId);return this.dtype==="string"?t:new Uint8Array(t.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),It().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(t=!1){return Ae.print(this,t)}clone(){return this.throwIfDisposed(),Ae.clone(this)}toString(t=!1){const n=this.dataSync();return vl(n,this.shape,this.dtype,t)}cast(t){return this.throwIfDisposed(),Ae.cast(this,t)}variable(t=!0,n,s){return this.throwIfDisposed(),It().makeVariable(this,t,n,s)}}Object.defineProperty(et,Symbol.hasInstance,{value:e=>!!e&&e.data!=null&&e.dataSync!=null&&e.throwIfDisposed!=null});function Sa(){return Rs("Tensor",()=>et)}Sa();class xn extends et{constructor(t,n,s,r){super(t.shape,t.dtype,t.dataId,r),this.trainable=n,this.name=s}assign(t){if(t.dtype!==this.dtype)throw new Error(`dtype of the new value (${t.dtype}) and previous value (${this.dtype}) must match`);if(!Pt(t.shape,this.shape))throw new Error(`shape of the new value (${t.shape}) and previous value (${this.shape}) must match`);It().disposeTensor(this),this.dataId=t.dataId,It().incRef(this,null)}dispose(){It().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(xn,Symbol.hasInstance,{value:e=>e instanceof et&&e.assign!=null&&e.assign instanceof Function});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Ir;(function(e){e.R0="R0",e.R1="R1",e.R2="R2",e.R3="R3",e.R4="R4",e.R5="R5",e.R6="R6"})(Ir||(Ir={}));var us;(function(e){e.float32="float32",e.int32="int32",e.bool="int32",e.complex64="complex64"})(us||(us={}));var cs;(function(e){e.float32="float32",e.int32="int32",e.bool="bool",e.complex64="complex64"})(cs||(cs={}));var ls;(function(e){e.float32="float32",e.int32="float32",e.bool="float32",e.complex64="complex64"})(ls||(ls={}));var hs;(function(e){e.float32="complex64",e.int32="complex64",e.bool="complex64",e.complex64="complex64"})(hs||(hs={}));const Al={float32:ls,int32:us,bool:cs,complex64:hs};function $a(e,t){if(e==="string"||t==="string"){if(e==="string"&&t==="string")return"string";throw new Error(`Can not upcast ${e} with ${t}`)}return Al[e][t]}function mN(e){return $a(e,"int32")}function ka(e){return e!=null&&typeof e=="object"&&"texture"in e&&e.texture instanceof WebGLTexture}function Ea(e){return typeof GPUBuffer<"u"&&e!=null&&typeof e=="object"&&"buffer"in e&&e.buffer instanceof GPUBuffer}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Q(e,t){if(e.dtype===t.dtype)return[e,t];const n=$a(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function va(e,t){g(e.dtype===t.dtype,()=>`The dtypes of the first(${e.dtype}) and second(${t.dtype}) input must match`)}function Dl(e,t){return t.some(n=>n.id===e.id)}function Vs(e){const t=[];return _a(e,t,new Set),t}function _a(e,t,n){if(e==null)return;if(e instanceof et){t.push(e);return}if(!Ol(e))return;const s=e;for(const r in s){const a=s[r];n.has(a)||(n.add(a),_a(a,t,n))}}function Ol(e){return Array.isArray(e)||typeof e=="object"}const dN=Object.freeze(Object.defineProperty({__proto__:null,assertTypesMatch:va,getTensorsInContainer:Vs,isTensorInList:Dl,makeTypesMatch:Q},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qn(e){return e.kernelName!=null}class Ar{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(t=>t.name)))}}}dispose(){for(const t in this.registeredVariables)this.registeredVariables[t].dispose()}}class je{constructor(t){this.ENV=t,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new Ar}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const s=t[n];if(await this.initializeBackend(s).success){await this.setBackend(s);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:t,asyncInit:n}=this.initializeBackendsAndReturnBest();if(n)throw new Error(`The highest priority backend '${t}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(t)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(t){if(!(t in this.registry))if(t in this.registryFactory){const{asyncInit:n}=this.initializeBackend(t);if(n)return null}else return null;return this.registry[t]}findBackendFactory(t){return t in this.registryFactory?this.registryFactory[t].factory:null}registerBackend(t,n,s=1){return t in this.registryFactory?(ue(`${t} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[t]={factory:n,priority:s},!0)}async setBackend(t){if(this.registryFactory[t]==null)throw new Error(`Backend name '${t}' not found in registry`);if(this.backendName=t,this.registry[t]==null){this.backendInstance=null;const{success:n,asyncInit:s}=this.initializeBackend(t);if(!(s?await n:n))return!1}return this.backendInstance=this.registry[t],this.setupRegisteredKernels(),this.profiler=new Tl(this.backendInstance),!0}setupRegisteredKernels(){ss(this.backendName).forEach(n=>{n.setupFunc!=null&&n.setupFunc(this.backendInstance)})}disposeRegisteredKernels(t){ss(t).forEach(s=>{s.disposeFunc!=null&&s.disposeFunc(this.registry[t])})}initializeBackend(t){const n=this.registryFactory[t];if(n==null)throw new Error(`Cannot initialize backend ${t}, no registration found.`);try{const s=n.factory();if(s&&!(s instanceof Jo)&&typeof s.then=="function"){const r=++this.pendingBackendInitId,a=s.then(o=>r<this.pendingBackendInitId?!1:(this.registry[t]=o,this.pendingBackendInit=null,!0)).catch(o=>(r<this.pendingBackendInitId||(this.pendingBackendInit=null,ue(`Initialization of backend ${t} failed`),ue(o.stack||o.message)),!1));return this.pendingBackendInit=a,{success:a,asyncInit:!0}}else return this.registry[t]=s,{success:!0,asyncInit:!1}}catch(s){return ue(`Initialization of backend ${t} failed`),ue(s.stack||s.message),{success:!1,asyncInit:!1}}}removeBackend(t){if(!(t in this.registryFactory))throw new Error(`${t} backend not found in registry`);this.backendName===t&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,t in this.registry&&(this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t]),delete this.registryFactory[t],this.backendName===t&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((t,n)=>this.registryFactory[n].priority-this.registryFactory[t].priority)}initializeBackendsAndReturnBest(){const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const s=t[n],{success:r,asyncInit:a}=this.initializeBackend(s);if(a||r)return{name:s,asyncInit:a}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(t,n){const s=this.state.tensorInfo.get(n),r=s.backend,a=this.readSync(n),o=r.refCount(n);r.disposeData(n,!0),s.backend=t,t.move(n,a,s.shape,s.dtype,o),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(t,n){let s=null;if(n==null){if(typeof t!="function")throw new Error("Please provide a function to tidy()");n=t}else{if(typeof t!="string"&&!(t instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof n!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");s=t}let r;return this.scopedRun(()=>this.startScope(s),()=>this.endScope(r),()=>(r=n(),r instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),r))}scopedRun(t,n,s){t();try{const r=s();return n(),r}catch(r){throw n(),r}}nextTensorId(){return je.nextTensorId++}nextVariableId(){return je.nextVariableId++}clone(t){const n=w.runKernel(la,{x:t}),s={x:t},r=o=>({x:()=>{const i="float32",u={x:o},c={dtype:i};return w.runKernel(ua,u,c)}}),a=[];return this.addTapeNode(this.state.activeScope.name,s,[n],r,a,{}),n}runKernel(t,n,s){if(this.backendName==null&&this.backend,!(vn(t,this.backendName)!=null))throw new Error(`Kernel '${t}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:t,inputs:n,attrs:s})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(t,n,s){const r=this.backend.numDataIds();let a=0;s.forEach(u=>{a+=u.dtype==="complex64"?3:1});const o=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],i=r-n-a-o;if(i>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${i} data ids) after running '${t}'`)}runKernelFunc(t){let n,s=[];const r=this.isTapeOn(),a=this.state.numBytes,o=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let i;this.backendName==null&&this.backend;let u;const c=Qn(t)?t.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(Qn(t)){const{kernelName:y,inputs:T,attrs:N}=t;this.backendName==null&&this.backend;const S=vn(y,this.backendName);g(S!=null,()=>`Cannot find registered kernel '${y}' for backend '${this.backendName}'`),i=()=>{const I=this.backend.numDataIds();u=S.kernelFunc({inputs:T,attrs:N,backend:this.backend});const A=Array.isArray(u)?u:[u];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(y,I,A);const k=A.map(_=>_.rank!=null?_:this.makeTensorFromTensorInfo(_));if(r){const _=this.getTensorsForGradient(y,T,k);s=this.saveTensorsForBackwardMode(_)}return k}}else{const{forwardFunc:y}=t,T=N=>{r&&(s=N.map(S=>this.keep(this.clone(S))))};i=()=>{const N=this.backend.numDataIds();u=this.tidy(()=>y(this.backend,T));const S=Array.isArray(u)?u:[u];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(c,N,S),S}}const{inputs:h,attrs:p}=t,f=Qn(t)?null:t.backwardsFunc;let d;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?n=i():(d=this.profiler.profileKernel(c,h,()=>i()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(d),n=d.outputs)}),r&&this.addTapeNode(c,h,n,f,s,p),this.state.profiling&&this.state.activeProfile.kernels.push({name:c,bytesAdded:this.state.numBytes-a,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-o,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(h).map(y=>h[y]!=null?h[y].shape:null),outputShapes:n.map(y=>y.shape),kernelTimeMs:d.timeMs,extraInfo:d.extraInfo}),Array.isArray(u)?n:n[0]}saveTensorsForBackwardMode(t){return t.map(s=>this.keep(this.clone(s)))}getTensorsForGradient(t,n,s){const r=Tr(t);if(r!=null){const a=r.inputsToSave||[],o=r.outputsToSave||[];let i;r.saveAllInputs?(g(Array.isArray(n),()=>"saveAllInputs is true, expected inputs to be an array."),i=Object.keys(n).map(c=>n[c])):i=a.map(c=>n[c]);const u=s.filter((c,h)=>o[h]);return i.concat(u)}return[]}makeTensor(t,n,s,r){if(t==null)throw new Error("Values passed to engine.makeTensor() are null");s=s||"float32",r=r||this.backend;let a=t;s==="string"&&Ln(t[0])&&(a=t.map(u=>zs(u)));const o=r.write(a,n,s),i=new et(n,s,o,this.nextTensorId());if(this.trackTensor(i,r),s==="string"){const u=this.state.tensorInfo.get(o),c=ea(a);this.state.numBytes+=c-u.bytes,u.bytes=c}return i}makeTensorFromDataId(t,n,s,r){s=s||"float32";const a={dataId:t,shape:n,dtype:s};return this.makeTensorFromTensorInfo(a,r)}makeTensorFromTensorInfo(t,n){const{dataId:s,shape:r,dtype:a}=t,o=new et(r,a,s,this.nextTensorId());return this.trackTensor(o,n),o}makeVariable(t,n=!0,s,r){s=s||this.nextVariableId().toString(),r!=null&&r!==t.dtype&&(t=t.cast(r));const a=new xn(t,n,s,this.nextTensorId());if(this.state.registeredVariables[a.name]!=null)throw new Error(`Variable with name ${a.name} was already registered`);return this.state.registeredVariables[a.name]=a,this.incRef(a,this.backend),a}trackTensor(t,n){this.state.numTensors++,t.dtype==="string"&&this.state.numStringTensors++;let s=0;t.dtype!=="complex64"&&t.dtype!=="string"&&(s=t.size*En(t.dtype)),this.state.numBytes+=s,this.state.tensorInfo.has(t.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(t.dataId,{backend:n||this.backend,dtype:t.dtype,shape:t.shape,bytes:s})),t instanceof xn||this.track(t)}incRef(t,n){this.trackTensor(t,n),this.backend.incRef(t.dataId)}removeDataId(t,n){this.state.tensorInfo.has(t)&&this.state.tensorInfo.get(t).backend===n&&(this.state.tensorInfo.delete(t),this.state.numDataBuffers--)}disposeTensor(t){if(!this.state.tensorInfo.has(t.dataId))return;const n=this.state.tensorInfo.get(t.dataId);if(this.state.numTensors--,t.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=n.bytes),t.dtype!=="complex64"&&t.dtype!=="string"){const s=t.size*En(t.dtype);this.state.numBytes-=s}n.backend.disposeData(t.dataId)&&this.removeDataId(t.dataId,n.backend)}disposeVariables(){for(const t in this.state.registeredVariables){const n=this.state.registeredVariables[t];this.disposeVariable(n)}}disposeVariable(t){this.disposeTensor(t),this.state.registeredVariables[t.name]!=null&&delete this.state.registeredVariables[t.name]}memory(){const t=this.backend.memory();return t.numTensors=this.state.numTensors,t.numDataBuffers=this.state.numDataBuffers,t.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(t.unreliable=!0,t.reasons==null&&(t.reasons=[]),t.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),t}async profile(t){this.state.profiling=!0;const n=this.state.numBytes,s=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await t(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(r=>r.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-n,this.state.activeProfile.newTensors=this.state.numTensors-s;for(const r of this.state.activeProfile.kernels)r.kernelTimeMs=await r.kernelTimeMs,r.extraInfo=await r.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(t,n,s,r,a,o){const i={id:this.state.nextTapeNodeId++,kernelName:t,inputs:n,outputs:s,saved:a},u=Tr(t);u!=null&&(r=u.gradFunc),r!=null&&(i.gradient=c=>(c=c.map((h,p)=>{if(h==null){const f=s[p],d=zn(f.size,f.dtype);return this.makeTensor(d,f.shape,f.dtype)}return h}),r(c.length>1?c:c[0],a,o))),this.state.activeTape.push(i)}keep(t){return t.kept=!0,t}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(t){const n={track:[],name:"unnamed scope",id:this.state.nextScopeId++};t&&(n.name=t),this.state.scopeStack.push(n),this.state.activeScope=n}endScope(t){const n=Vs(t),s=new Set(n.map(a=>a.id));for(let a=0;a<this.state.activeScope.track.length;a++){const o=this.state.activeScope.track[a];!o.kept&&!s.has(o.id)&&o.dispose()}const r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],n.forEach(a=>{!a.kept&&a.scopeId===r.id&&this.track(a)})}gradients(t,n,s,r=!1){if(g(n.length>0,()=>"gradients() received an empty list of xs."),s!=null&&s.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${s.dtype}'`);const a=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",t));g(a instanceof et,()=>"The result y returned by f() must be a tensor.");const o=kl(this.state.activeTape,n,a);if(!r&&o.length===0&&n.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const i={};i[a.id]=s??Fl(a.shape),El(i,o,c=>this.tidy(c),Cl);const u=n.map(c=>i[c.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(c=>{for(const h of c.saved)h.dispose()}),this.state.activeTape=null),{value:a,grads:u}})}customGrad(t){return g(Qt(t),()=>"The f passed in customGrad(f) must be a function."),(...n)=>{g(n.every(i=>i instanceof et),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let s;const r={};n.forEach((i,u)=>{r[u]=i});const a=(i,u)=>(s=t(...n,u),g(s.value instanceof et,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),g(Qt(s.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),s.value),o=(i,u)=>{const c=s.gradFunc(i,u),h=Array.isArray(c)?c:[c];g(h.length===n.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),g(h.every(f=>f instanceof et),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const p={};return h.forEach((f,d)=>{p[d]=()=>f}),p};return this.runKernelFunc({forwardFunc:a,backwardsFunc:o,inputs:r})}}readSync(t){return this.state.tensorInfo.get(t).backend.readSync(t)}read(t){return this.state.tensorInfo.get(t).backend.read(t)}readToGPU(t,n){return this.state.tensorInfo.get(t).backend.readToGPU(t,n)}async time(t){const n=tn(),s=await this.backend.time(t);return s.wallMs=tn()-n,s}track(t){return this.state.activeScope!=null&&(t.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(t)),t}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new Ar;for(const t in this.registry)this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}je.nextTensorId=0;je.nextVariableId=0;function Fl(e){const t=Bs(Z(e),"float32");return w.makeTensor(t,e,"float32")}function xa(){const e=oa();if(e._tfengine==null){const t=new di(e);e._tfengine=new je(t)}return wi(e._tfengine.ENV),xl(()=>e._tfengine),e._tfengine}const w=xa();function Cl(e,t){const n={a:e,b:t};return w.runKernel(ia,n)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bl(){return typeof navigator<"u"&&navigator!=null}let ps;function Rl(e){ps=e}function Pl(e){if(ps!==void 0)return ps;if(e||Bl()){if(e||(e=navigator),e.product==="ReactNative")return!0;const t=e.userAgent||e.vendor||(typeof window<"u"?window.opera:"");if(!t){const n=e;return n.userAgentData&&n.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function Ia(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}const gN=Object.freeze(Object.defineProperty({__proto__:null,isBrowser:Ia,isMobile:Pl,mockIsMobile:Rl},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ft=C();ft.registerFlag("DEBUG",()=>!1,e=>{e&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});ft.registerFlag("IS_BROWSER",()=>Ia());ft.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");ft.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));ft.registerFlag("IS_SAFARI",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor));ft.registerFlag("PROD",()=>!1);ft.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>ft.getBool("DEBUG"));ft.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);ft.registerFlag("IS_TEST",()=>!1);ft.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>ft.getBool("DEBUG"));ft.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);ft.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);ft.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function se(e,t){let n=e;if(St(e))return t==="string"?[]:[e.length];if(ka(e)){const r=e.channels||"RGBA";return[e.height,e.width*r.length]}else if(Ea(e))return[e.buffer.size/(t==null?4:En(t))];if(!Array.isArray(e))return[];const s=[];for(;Array.isArray(n)||St(n)&&t!=="string";)s.push(n.length),n=n[0];return Array.isArray(e)&&C().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&Aa(e,s,[]),s}function Aa(e,t,n){if(n=n||[],!Array.isArray(e)&&!St(e)){g(t.length===0,()=>`Element arr[${n.join("][")}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}g(t.length>0,()=>`Element arr[${n.join("][")}] should be a primitive, but is an array of ${e.length} elements`),g(e.length===t[0],()=>`Element arr[${n.join("][")}] should have ${t[0]} elements, but has ${e.length} elements`);const s=t.slice(1);for(let r=0;r<e.length;++r)Aa(e[r],s,n.concat(r))}function Dr(e,t,n,s){if(e!=="string_or_numeric"){if(e==null)throw new Error("Expected dtype cannot be null.");if(e!=="numeric"&&e!==t||e==="numeric"&&t==="string")throw new Error(`Argument '${n}' passed to '${s}' must be ${e} tensor, but got ${t} tensor`)}}function m(e,t,n,s="numeric"){if(e instanceof Sa())return Dr(s,e.dtype,t,n),e;let r=hn(e);if(r!=="string"&&["bool","int32","float32"].indexOf(s)>=0&&(r=s),Dr(s,r,t,n),e==null||!St(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string"){const u=e==null?"null":e.constructor.name;throw new Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${u}'`)}const a=se(e,r);!St(e)&&!Array.isArray(e)&&(e=[e]);const i=r!=="string"?Vn(e,r):Ue(e,[],!0);return w.makeTensor(i,a,r)}function en(e,t,n,s="numeric"){if(!Array.isArray(e))throw new Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((a,o)=>m(a,`${t}[${o}]`,n,s))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Da="__op";function b(e){const t=Object.keys(e);if(t.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0];const s=e[n];n.endsWith("_")&&(n=n.substring(0,n.length-1)),n=n+Da;const r=(...a)=>{w.startScope(n);try{const o=s(...a);return te(o)&&console.error("Cannot return a Promise inside of tidy."),w.endScope(o),o}catch(o){throw w.endScope(null),o}};return Object.defineProperty(r,"name",{value:n,configurable:!0}),r}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ll(e,t){const n=m(e,"real","complex"),s=m(t,"imag","complex");lt(n.shape,s.shape,`real and imag shapes, ${n.shape} and ${s.shape}, must match in call to tf.complex().`);const r={real:n,imag:s};return w.runKernel(Wi,r)}const ee=b({complex_:Ll});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function re(e,t,n,s){if(s==null)s=hn(e);else if(s==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(Ea(e)||ka(e)){if(s!=="float32"&&s!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${s}.`);return w.backend.createTensorFromGPUData(e,t||n,s)}if(!St(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(t!=null){bt(t);const r=Z(t),a=Z(n);g(r===a,()=>`Based on the provided shape, [${t}], the tensor should have ${r} values but has ${a}`);for(let o=0;o<n.length;++o){const i=n[o],u=o===n.length-1?i!==Z(t.slice(o)):!0;g(n[o]===t[o]||!u,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return!St(e)&&!Array.isArray(e)&&(e=[e]),t=t||n,e=s!=="string"?Vn(e,s):Ue(e,[],!0),w.makeTensor(e,t,s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bt(e,t,n){const s=se(e,n);return re(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const de={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};class Ot{static join(t){return new Ot(t).slice()}constructor(t){if(this.shards=[],this.previousShardIndex=0,t==null||(t instanceof Array||(t=[t]),t=t.map(s=>St(s)?s.buffer:s),t.length===0))return;this.bufferUniformSize=t[0].byteLength;let n=0;for(let s=0;s<t.length;s++){const r=t[s];s!==t.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const a=n+r.byteLength;this.shards.push({buffer:r,start:n,end:a}),n=a}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(t=0,n=this.byteLength){if(this.shards.length===0)return new ArrayBuffer(0);if(t=isNaN(Number(t))?0:t,n=isNaN(Number(n))?0:n,t=Math.max(0,t),n=Math.min(this.byteLength,n),n<=t)return new ArrayBuffer(0);const s=this.findShardForByte(t);if(s===-1)throw new Error(`Could not find start shard for byte ${t}`);const r=n-t,a=new ArrayBuffer(r),o=new Uint8Array(a);let i=0;for(let u=s;u<this.shards.length;u++){const c=this.shards[u],p=t+i-c.start,f=i,y=Math.min(n,c.end)-c.start,T=new Uint8Array(c.buffer,p,y-p);if(o.set(T,f),i+=T.length,n<c.end)break}return a}findShardForByte(t){if(this.shards.length===0||t<0||t>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(t/this.bufferUniformSize),this.previousShardIndex;function n(r){return t<r.start?-1:t>=r.end?1:0}if(n(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;const s=zl(this.shards,n);return s===-1?-1:(this.previousShardIndex=s,this.previousShardIndex)}}function zl(e,t){let n=0,s=e.length;for(;n<=s;){const r=Math.floor((s-n)/2)+n,a=t(e[r]);if(a===0)return r;a<0?s=r:n=r+1}return-1}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yN(){C().set("PROD",!0)}function bN(){C().set("DEBUG",!0)}function wN(){C().set("DEPRECATION_WARNINGS_ENABLED",!1),console.warn("TensorFlow.js deprecation warnings have been disabled.")}function NN(e){C().getBool("DEPRECATION_WARNINGS_ENABLED")&&console.warn(e+" You can disable deprecation warnings with tf.disableDeprecationWarnings().")}function TN(){w.disposeVariables()}function SN(){return w}function $N(){return w.memory()}function kN(e){return w.profile(e)}function V(e,t){return w.tidy(e,t)}function pt(e){Vs(e).forEach(n=>n.dispose())}function Ct(e){return w.keep(e)}function EN(e){return w.time(e)}function vN(e){return w.setBackend(e)}function _N(){return w.ready()}function Vl(){return w.backendName}function xN(e){w.removeBackend(e)}function IN(e){return w.findBackend(e)}function AN(e){return w.findBackendFactory(e)}function DN(e,t,n=1){return w.registerBackend(e,t,n)}function Ul(){return w.backend}function ON(e,t){C().setPlatform(e,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ne=4;async function jl(e,t){const n=[],s=[],r=Array.isArray(e)?e.map(o=>o.name):Object.keys(e);for(let o=0;o<r.length;++o){const i=r[o],u=Array.isArray(e)?e[o].tensor:e[i];if(u.dtype!=="float32"&&u.dtype!=="int32"&&u.dtype!=="bool"&&u.dtype!=="string"&&u.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${i}': ${u.dtype}`);const c={name:i,shape:u.shape,dtype:u.dtype};if(u.dtype==="string"){const h=new Promise(async p=>{const f=await u.bytes(),d=f.reduce((N,S)=>N+S.length,0)+ne*f.length,y=new Uint8Array(d);let T=0;for(let N=0;N<f.length;N++){const S=f[N],I=new Uint8Array(new Uint32Array([S.length]).buffer);y.set(I,T),T+=ne,y.set(S,T),T+=S.length}p(y)});s.push(h)}else s.push(u.data());t!=null&&(c.group=t),n.push(c)}const a=await Promise.all(s);return{data:Gl(a),specs:n}}function Oa(e,t){const n=new Ot(e),s={};let r=0;for(const a of t){const o=Wl(a,(i,u)=>n.slice(r+i,r+u));s[a.name]=Fa(a,n.slice(r,r+o)),r+=o}return s}function Wl(e,t){const n=Z(e.shape);let s;if("quantization"in e){const r=e.quantization;s=de[r.dtype]}else if(e.dtype==="string"){let r=0;for(let a=0;a<n;a++)r+=ne+new Uint32Array(t(r,r+ne))[0];return r}else s=de[e.dtype];return n*s}async function ql(e,t){const n=Z(e.shape);let s;if("quantization"in e){const r=e.quantization;s=de[r.dtype]}else if(e.dtype==="string"){let r=0;for(let a=0;a<n;a++)r+=ne+new Uint32Array(await t(r,r+ne))[0];return r}else s=de[e.dtype];return n*s}function Fa(e,t){const n=e.name,s=e.dtype,r=e.shape,a=Z(r);let o,i=0;if("quantization"in e){const u=e.quantization;if(u.dtype==="uint8"||u.dtype==="uint16"){if(!("min"in u&&"scale"in u))throw new Error(`Weight ${e.name} with quantization ${u.dtype} doesn't have corresponding metadata min and scale.`)}else if(u.dtype==="float16"){if(s!=="float32")throw new Error(`Weight ${e.name} is quantized with ${u.dtype} which only supports weights of type float32 not ${s}.`)}else throw new Error(`Weight ${e.name} has unknown quantization dtype ${u.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);const c=de[u.dtype],h=u.dtype==="uint8"?new Uint8Array(t):new Uint16Array(t);if(s==="float32")if(u.dtype==="uint8"||u.dtype==="uint16"){o=new Float32Array(h.length);for(let p=0;p<h.length;p++){const f=h[p];o[p]=f*u.scale+u.min}}else if(u.dtype==="float16")o=Zl()(h);else throw new Error(`Unsupported quantization type ${u.dtype} for weight type float32.`);else if(s==="int32"){if(u.dtype!=="uint8"&&u.dtype!=="uint16")throw new Error(`Unsupported quantization type ${u.dtype} for weight type int32.`);o=new Int32Array(h.length);for(let p=0;p<h.length;p++){const f=h[p];o[p]=Math.round(f*u.scale+u.min)}}else throw new Error(`Unsupported dtype in weight '${n}': ${s}`);i+=a*c}else if(s==="string"){const u=Z(e.shape);o=[];for(let c=0;c<u;c++){const h=new Uint32Array(t.slice(i,i+ne))[0];i+=ne;const p=new Uint8Array(t.slice(i,i+h));o.push(p),i+=h}}else{const u=de[s];if(s==="float32")o=new Float32Array(t);else if(s==="int32")o=new Int32Array(t);else if(s==="bool")o=new Uint8Array(t);else if(s==="complex64"){o=new Float32Array(t);const c=new Float32Array(o.length/2),h=new Float32Array(o.length/2);for(let y=0;y<c.length;y++)c[y]=o[y*2],h[y]=o[y*2+1];const p=Bt(c,r,"float32"),f=Bt(h,r,"float32"),d=ee(p,f);return p.dispose(),f.dispose(),d}else throw new Error(`Unsupported dtype in weight '${n}': ${s}`);i+=a*u}return Bt(o,r,s)}async function Or(e,t,n){let s=new Uint8Array(t);for(;s.byteLength<n;){const{done:r,value:a}=await e.read();if(r&&a==null){const i=n-s.byteLength;throw new Error(`Reader is done but ${i} bytes are still expected`)}const o=new Uint8Array(s.length+a.byteLength);o.set(s,0),o.set(new Uint8Array(a),s.length),s=o}return s.buffer}async function Ca(e,t){const n={},s=e.getReader();let r=new ArrayBuffer(0);for(const a of t){const o=await ql(a,async(c,h)=>(r=await Or(s,r,h),r.slice(c,h)));r=await Or(s,r,o);const i=r.slice(0,o);r=r.slice(o);const u=Fa(a,i);if(n[a.name]=u,Vl()==="webgpu"){const c=Ul();"uploadToGPU"in c&&Z(u.shape)>=C().get("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD")&&c.uploadToGPU(u.dataId)}}return n}function Gl(e){if(e===null)throw new Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0;const n=[];e.forEach(a=>{if(t+=a.byteLength,n.push(a.byteLength===a.buffer.byteLength?a:new a.constructor(a)),!(a instanceof Float32Array||a instanceof Int32Array||a instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${a.constructor.name}`)});const s=new Uint8Array(t);let r=0;return n.forEach(a=>{s.set(new Uint8Array(a.buffer),r),r+=a.byteLength}),s.buffer}const Us=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function Fr(e){return Us?Buffer.byteLength(e,"utf8"):new Blob([e]).size}function Ml(e){if(Us)return Buffer.from(e).toString("base64");const t=new Uint8Array(e);let n="";for(let s=0,r=t.length;s<r;s++)n+=String.fromCharCode(t[s]);return btoa(n)}function Kl(e){if(Us){const s=Buffer.from(e,"base64");return s.buffer.slice(s.byteOffset,s.byteOffset+s.byteLength)}const t=atob(e),n=new Uint8Array(t.length);for(let s=0;s<t.length;++s)n.set([t.charCodeAt(s)],s);return n.buffer}function Hl(e){return Ot.join(e)}function Cr(e){const t="/";for(e=e.trim();e.endsWith(t);)e=e.slice(0,e.length-1);const n=e.split(t);return n[n.length-1]}function Ba(e,t){const n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return e.signature!=null&&(n.signature=e.signature),e.userDefinedMetadata!=null&&(n.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(n.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(n.initializerSignature=e.initializerSignature),e.trainingConfig!=null&&(n.trainingConfig=e.trainingConfig),n}function js(e,t,n){const s={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(e.trainingConfig!=null&&(s.trainingConfig=e.trainingConfig),e.weightsManifest!=null){if(!t)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!n)throw new Error("modelJSON has weightsManifest but weightData is null");s.weightSpecs=t,s.weightData=n}return e.signature!=null&&(s.signature=e.signature),e.userDefinedMetadata!=null&&(s.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(s.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(s.initializerSignature=e.initializerSignature),s}async function Ws(e,t){let n,s;return e.weightsManifest!=null&&([n,s]=await t(e.weightsManifest)),js(e,n,s)}function mn(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:e.modelTopology==null?0:Fr(JSON.stringify(e.modelTopology)),weightSpecsBytes:e.weightSpecs==null?0:Fr(JSON.stringify(e.weightSpecs)),weightDataBytes:e.weightData==null?0:new Ot(e.weightData).byteLength}}function In(e){const t=[];for(const n of e)t.push(...n.weights);return t}function Xl(){const e=n=>{let s=n<<13,r=0;for(;!(s&8388608);)r-=8388608,s<<=1;return s&=-8388609,r+=947912704,s|r},t=new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let n=1024;n<2048;n++)t[n]=939524096+(n-1024<<13);return t}function Jl(){const e=new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let t=33;t<63;t++)e[t]=2147483648+(t-32<<23);return e}function Yl(){const e=new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}function Zl(){const e=Xl(),t=Jl(),n=Yl();return s=>{const r=new ArrayBuffer(4*s.length),a=new Uint32Array(r);for(let o=0;o<s.length;o++){const i=s[o],u=e[n[i>>10]+(i&1023)]+t[i>>10];a[o]=u}return new Float32Array(r)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Y{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return Y.instance==null&&(Y.instance=new Y),Y.instance}static registerSaveRouter(t){Y.getInstance().saveRouters.push(t)}static registerLoadRouter(t){Y.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return Y.getHandlers(t,"save")}static getLoadHandlers(t,n){return Y.getHandlers(t,"load",n)}static getHandlers(t,n,s){const r=[];return(n==="load"?Y.getInstance().loadRouters:Y.getInstance().saveRouters).forEach(o=>{const i=o(t,s);i!==null&&r.push(i)}),r}}const Ql=e=>Y.registerSaveRouter(e),th=e=>Y.registerLoadRouter(e),eh=e=>Y.getSaveHandlers(e),nh=(e,t)=>Y.getLoadHandlers(e,t);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fs="tensorflowjs",ms=1,pe="models_store",Xt="model_info_store";function Ra(){if(!C().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const e=typeof window>"u"?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(t==null)throw new Error("The current browser does not appear to support IndexedDB.");return t}function ds(e){const t=e.result;t.createObjectStore(pe,{keyPath:"modelPath"}),t.createObjectStore(Xt,{keyPath:"modelPath"})}class ge{constructor(t){if(this.indexedDB=Ra(),t==null||!t)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=t}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,t)}async load(){return this.databaseAction(this.modelPath)}databaseAction(t,n){return new Promise((s,r)=>{const a=this.indexedDB.open(fs,ms);a.onupgradeneeded=()=>ds(a),a.onsuccess=()=>{const o=a.result;if(n==null){const i=o.transaction(pe,"readonly"),c=i.objectStore(pe).get(this.modelPath);c.onsuccess=()=>{if(c.result==null)return o.close(),r(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));s(c.result.modelArtifacts)},c.onerror=h=>(o.close(),r(c.error)),i.oncomplete=()=>o.close()}else{n.weightData=Ot.join(n.weightData);const i=mn(n),u=o.transaction(Xt,"readwrite");let c=u.objectStore(Xt),h;try{h=c.put({modelPath:this.modelPath,modelArtifactsInfo:i})}catch(f){return r(f)}let p;h.onsuccess=()=>{p=o.transaction(pe,"readwrite");const f=p.objectStore(pe);let d;try{d=f.put({modelPath:this.modelPath,modelArtifacts:n,modelArtifactsInfo:i})}catch(y){return r(y)}d.onsuccess=()=>s({modelArtifactsInfo:i}),d.onerror=y=>{c=u.objectStore(Xt);const T=c.delete(this.modelPath);T.onsuccess=()=>(o.close(),r(d.error)),T.onerror=N=>(o.close(),r(d.error))}},h.onerror=f=>(o.close(),r(h.error)),u.oncomplete=()=>{p==null?o.close():p.oncomplete=()=>o.close()}}},a.onerror=o=>r(a.error)})}}ge.URL_SCHEME="indexeddb://";const Pa=e=>C().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(ge.URL_SCHEME)?sh(e.slice(ge.URL_SCHEME.length)):null;Y.registerSaveRouter(Pa);Y.registerLoadRouter(Pa);function sh(e){return new ge(e)}function rh(e){return e.startsWith(ge.URL_SCHEME)?e.slice(ge.URL_SCHEME.length):e}class ah{constructor(){this.indexedDB=Ra()}async listModels(){return new Promise((t,n)=>{const s=this.indexedDB.open(fs,ms);s.onupgradeneeded=()=>ds(s),s.onsuccess=()=>{const r=s.result,a=r.transaction(Xt,"readonly"),i=a.objectStore(Xt).getAll();i.onsuccess=()=>{const u={};for(const c of i.result)u[c.modelPath]=c.modelArtifactsInfo;t(u)},i.onerror=u=>(r.close(),n(i.error)),a.oncomplete=()=>r.close()},s.onerror=r=>n(s.error)})}async removeModel(t){return t=rh(t),new Promise((n,s)=>{const r=this.indexedDB.open(fs,ms);r.onupgradeneeded=()=>ds(r),r.onsuccess=()=>{const a=r.result,o=a.transaction(Xt,"readwrite"),i=o.objectStore(Xt),u=i.get(t);let c;u.onsuccess=()=>{if(u.result==null)return a.close(),s(new Error(`Cannot find model with path '${t}' in IndexedDB.`));{const h=i.delete(t),p=()=>{c=a.transaction(pe,"readwrite");const d=c.objectStore(pe).delete(t);d.onsuccess=()=>n(u.result.modelArtifactsInfo),d.onerror=y=>s(u.error)};h.onsuccess=p,h.onerror=f=>(p(),a.close(),s(u.error))}},u.onerror=h=>(a.close(),s(u.error)),o.oncomplete=()=>{c==null?a.close():c.oncomplete=()=>a.close()}},r.onerror=a=>s(r.error)})}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ut="/",De="tensorflowjs_models",La="info",oh="model_topology",ih="weight_specs",uh="weight_data",ch="model_metadata";function za(e){return{info:[De,e,La].join(Ut),topology:[De,e,oh].join(Ut),weightSpecs:[De,e,ih].join(Ut),weightData:[De,e,uh].join(Ut),modelMetadata:[De,e,ch].join(Ut)}}function Va(e){for(const t of Object.values(e))window.localStorage.removeItem(t)}function lh(e){const t=e.split(Ut);if(t.length<3)throw new Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(Ut)}function hh(e){return e.startsWith(ye.URL_SCHEME)?e.slice(ye.URL_SCHEME.length):e}class ye{constructor(t){if(!C().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,t==null||!t)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=t,this.keys=za(this.modelPath)}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const n=JSON.stringify(t.modelTopology),s=JSON.stringify(t.weightSpecs),r=mn(t),a=Ot.join(t.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,n),this.LS.setItem(this.keys.weightSpecs,s),this.LS.setItem(this.keys.weightData,Ml(a));const o={format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,signature:t.signature!=null?t.signature:void 0,userDefinedMetadata:t.userDefinedMetadata!=null?t.userDefinedMetadata:void 0,modelInitializer:t.modelInitializer!=null?t.modelInitializer:void 0,initializerSignature:t.initializerSignature!=null?t.initializerSignature:void 0,trainingConfig:t.trainingConfig!=null?t.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(o)),{modelArtifactsInfo:r}}catch{throw Va(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){const t=JSON.parse(this.LS.getItem(this.keys.info));if(t==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(t.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const n={},s=JSON.parse(this.LS.getItem(this.keys.topology));if(s==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);n.modelTopology=s;const r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);n.weightSpecs=r;const a=this.LS.getItem(this.keys.modelMetadata);if(a!=null){const i=JSON.parse(a);n.format=i.format,n.generatedBy=i.generatedBy,n.convertedBy=i.convertedBy,i.signature!=null&&(n.signature=i.signature),i.userDefinedMetadata!=null&&(n.userDefinedMetadata=i.userDefinedMetadata),i.modelInitializer!=null&&(n.modelInitializer=i.modelInitializer),i.initializerSignature!=null&&(n.initializerSignature=i.initializerSignature),i.trainingConfig!=null&&(n.trainingConfig=i.trainingConfig)}const o=this.LS.getItem(this.keys.weightData);if(o==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return n.weightData=Kl(o),n}}ye.URL_SCHEME="localstorage://";const Ua=e=>C().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(ye.URL_SCHEME)?ph(e.slice(ye.URL_SCHEME.length)):null;Y.registerSaveRouter(Ua);Y.registerLoadRouter(Ua);function ph(e){return new ye(e)}class fh{constructor(){g(C().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),g(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const t={},n=De+Ut,s=Ut+La;for(let r=0;r<this.LS.length;++r){const a=this.LS.key(r);if(a.startsWith(n)&&a.endsWith(s)){const o=lh(a);t[o]=JSON.parse(this.LS.getItem(a))}}return t}async removeModel(t){t=hh(t);const n=za(t);if(this.LS.getItem(n.info)==null)throw new Error(`Cannot find model at path '${t}'`);const s=JSON.parse(this.LS.getItem(n.info));return Va(n),s}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ce="://";class it{constructor(){this.managers={}}static getInstance(){return it.instance==null&&(it.instance=new it),it.instance}static registerManager(t,n){g(t!=null,()=>"scheme must not be undefined or null."),t.endsWith(Ce)&&(t=t.slice(0,t.indexOf(Ce))),g(t.length>0,()=>"scheme must not be an empty string.");const s=it.getInstance();g(s.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),s.managers[t]=n}static getManager(t){const n=it.getInstance().managers[t];if(n==null)throw new Error(`Cannot find model manager for scheme '${t}'`);return n}static getSchemes(){return Object.keys(it.getInstance().managers)}}function Tn(e){if(e.indexOf(Ce)===-1)throw new Error(`The url string provided does not contain a scheme. Supported schemes are: ${it.getSchemes().join(",")}`);return{scheme:e.split(Ce)[0],path:e.split(Ce)[1]}}async function ja(e,t,n=!1){g(e!==t,()=>`Old path and new path are the same: '${e}'`);const s=Y.getLoadHandlers(e);g(s.length>0,()=>`Copying failed because no load handler is found for source URL ${e}.`),g(s.length<2,()=>`Copying failed because more than one (${s.length}) load handlers for source URL ${e}.`);const r=s[0],a=Y.getSaveHandlers(t);g(a.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),g(a.length<2,()=>`Copying failed because more than one (${s.length}) save handlers for destination URL ${t}.`);const o=a[0],i=Tn(e).scheme,u=Tn(e).path,c=i===Tn(e).scheme,h=await r.load();n&&c&&await it.getManager(i).removeModel(u);const p=await o.save(h);return n&&!c&&await it.getManager(i).removeModel(u),p.modelArtifactsInfo}async function mh(){const e=it.getSchemes(),t={};for(const n of e){const s=await it.getManager(n).listModels();for(const r in s){const a=n+Ce+r;t[a]=s[r]}}return t}async function dh(e){const t=Tn(e);return it.getManager(t.scheme).removeModel(t.path)}async function gh(e,t){return ja(e,t,!1)}async function yh(e,t){return ja(e,t,!0)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class bh{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(t,n){return fetch(t,n)}now(){return performance.now()}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${n}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(t)}decode(t,n){return new TextDecoder(n).decode(t)}setTimeoutCustom(t,n){if(typeof window>"u"||!C().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(t,n);return}this.functionRefs.push(t),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},n),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",s=>{if(s.source===window&&s.data.name===this.messageName){s.stopPropagation();const r=this.functionRefs[s.data.index];r(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(t){return pa(t)}}if(C().get("IS_BROWSER")){C().setPlatform("browser",new bh);try{it.registerManager(ye.URL_SCHEME,new fh)}catch{}try{it.registerManager(ge.URL_SCHEME,new ah)}catch{}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wh={importFetch:()=>require("node-fetch")};let ts;class Nh{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(t,n){return C().global.fetch!=null?C().global.fetch(t,n):(ts==null&&(ts=wh.importFetch()),ts(t,n))}now(){const t=process.hrtime();return t[0]*1e3+t[1]/1e6}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${n}`);return this.textEncoder.encode(t)}decode(t,n){return t.length===0?"":new this.util.TextDecoder(n).decode(t)}isTypedArray(t){return this.util.types.isFloat32Array(t)||this.util.types.isInt32Array(t)||this.util.types.isUint8Array(t)||this.util.types.isUint8ClampedArray(t)}}C().get("IS_NODE")&&!C().get("IS_BROWSER")&&C().setPlatform("node",new Nh);/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jt(e,t="float32",n){return t=t||"float32",bt(e),new is(e,t,n)}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Th(e,t){const n=m(e,"x","cast");if(!ta(t))throw new Error(`Failed to cast to unknown dtype ${t}`);if(t==="string"&&n.dtype!=="string"||t!=="string"&&n.dtype==="string")throw new Error("Only strings can be casted to strings");const s={x:n},r={dtype:t};return w.runKernel(ua,s,r)}const tt=b({cast_:Th});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sh(e){const n={x:m(e,"x","clone","string_or_numeric")};return w.runKernel(la,n)}const Yt=b({clone_:Sh});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wa(e,t=!1){console.log(e.toString(t))}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */xa();const $h={buffer:jt,cast:tt,clone:Yt,print:Wa};Il($h);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kh(e,t){let n=m(e,"a","add"),s=m(t,"b","add");[n,s]=Q(n,s);const r={a:n,b:s};return w.runKernel(ia,r)}const F=b({add_:kh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Eh(e,t){let n=m(e,"a","floorDiv"),s=m(t,"b","floorDiv");[n,s]=Q(n,s);const r={a:n,b:s};return w.runKernel(Tu,r)}const qa=b({floorDiv_:Eh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vh(e,t){let n=m(e,"a","div"),s=m(t,"b","div");if([n,s]=Q(n,s),n.dtype==="int32"&&s.dtype==="int32")return qa(n,s);const r={a:n,b:s},a={};return w.runKernel(cu,r,a)}const K=b({div_:vh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _h(e,t){let n=m(e,"a","mul"),s=m(t,"b","mul");[n,s]=Q(n,s);const r={a:n,b:s};return w.runKernel(Qu,r)}const v=b({mul_:_h});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xh(e){const t=m(e,"x","abs");if(t.dtype==="complex64"){const n={x:t};return w.runKernel(qi,n)}else{const n={x:t};return w.runKernel(Ti,n)}}const Nt=b({abs_:xh});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ih(e){const n={x:m(e,"x","acos")};return w.runKernel(Si,n)}const Ah=b({acos_:Ih});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dh(e){const n={x:m(e,"x","acosh")};return w.runKernel($i,n)}const Oh=b({acosh_:Dh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fh(e){g(Array.isArray(e),()=>"The argument passed to tf.addN() must be a list of tensors"),g(e.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${e.length}`);const t=e.map((r,a)=>m(r,`tensors${a}`,"addN")),n=t[0];t.forEach(r=>{if(r.dtype!==n.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),t.forEach(r=>{if(!Pt(r.shape,n.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const s=t;return w.runKernel(ki,s)}const Ch=b({addN_:Fh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bh(e,t=null,n=!1){const r={x:m(e,"x","all","bool")},a={axis:t,keepDims:n};return w.runKernel(Ei,r,a)}const Rh=b({all_:Bh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ph(e,t=null,n=!1){const r={x:m(e,"x","any","bool")},a={axis:t,keepDims:n};return w.runKernel(vi,r,a)}const Lh=b({any_:Ph});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zh(e,t=0){const s={x:m(e,"x","argMax")},r={axis:t};return w.runKernel(_i,s,r)}const Vh=b({argMax_:zh});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uh(e,t=0){const s={x:m(e,"x","argMin")},r={axis:t};return w.runKernel(xi,s,r)}const jh=b({argMin_:Uh});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wh(e){const n={x:m(e,"x","asin")};return w.runKernel(Ii,n)}const qh=b({asin_:Wh});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gh(e){const n={x:m(e,"x","asinh")};return w.runKernel(Ai,n)}const Mh=b({asinh_:Gh});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kh(e){const n={x:m(e,"x","atan")};return w.runKernel(Di,n)}const Hh=b({atan_:Kh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xh(e,t){let n=m(e,"a","atan2"),s=m(t,"b","atan2");[n,s]=Q(n,s);const r={a:n,b:s};return w.runKernel(Fi,r)}const Jh=b({atan2_:Xh});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yh(e){const n={x:m(e,"x","atanh")};return w.runKernel(Oi,n)}const Zh=b({atanh_:Yh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function FN(e,t,n,s,r="NHWC",a){const o=e[3],i=[...t,o],u=ap(r);return Un(e,i,n,a,s,null,null,u)}function Qh(e,t,n,s,r,a,o="channelsLast"){const[i,u]=nn(t);let c;if(o==="channelsLast")c=[i,u,e[3],e[3]];else if(o==="channelsFirst")c=[i,u,e[1],e[1]];else throw new Error(`Unknown dataFormat ${o}`);return Un(e,c,n,s,r,a,!1,o)}function CN(e,t,n,s,r,a,o="NDHWC"){const[i,u,c]=gs(t);let h,p;if(o==="NDHWC")p="channelsLast",h=[i,u,c,e[4],e[4]];else if(o==="NCDHW")p="channelsFirst",h=[i,u,c,e[1],e[1]];else throw new Error(`Unknown dataFormat ${o}`);return tp(e,h,n,s,r,!1,p,a)}function Un(e,t,n,s,r,a,o=!1,i="channelsLast"){let[u,c,h,p]=[-1,-1,-1,-1];if(i==="channelsLast")[u,c,h,p]=e;else if(i==="channelsFirst")[u,p,c,h]=e;else throw new Error(`Unknown dataFormat ${i}`);const[f,d,,y]=t,[T,N]=nn(n),[S,I]=nn(s),A=Be(f,S),k=Be(d,I),{padInfo:_,outHeight:x,outWidth:D}=sp(r,c,h,T,N,A,k,a,i),O=o?y*p:y;let B;return i==="channelsFirst"?B=[u,O,x,D]:i==="channelsLast"&&(B=[u,x,D,O]),{batchSize:u,dataFormat:i,inHeight:c,inWidth:h,inChannels:p,outHeight:x,outWidth:D,outChannels:O,padInfo:_,strideHeight:T,strideWidth:N,filterHeight:f,filterWidth:d,effectiveFilterHeight:A,effectiveFilterWidth:k,dilationHeight:S,dilationWidth:I,inShape:e,outShape:B,filterShape:t}}function tp(e,t,n,s,r,a=!1,o="channelsLast",i){let[u,c,h,p,f]=[-1,-1,-1,-1,-1];if(o==="channelsLast")[u,c,h,p,f]=e;else if(o==="channelsFirst")[u,f,c,h,p]=e;else throw new Error(`Unknown dataFormat ${o}`);const[d,y,T,,N]=t,[S,I,A]=gs(n),[k,_,x]=gs(s),D=Be(d,k),O=Be(y,_),B=Be(T,x),{padInfo:L,outDepth:R,outHeight:G,outWidth:j}=rp(r,c,h,p,S,I,A,D,O,B,i),J=a?N*f:N;let st;return o==="channelsFirst"?st=[u,J,R,G,j]:o==="channelsLast"&&(st=[u,R,G,j,J]),{batchSize:u,dataFormat:o,inDepth:c,inHeight:h,inWidth:p,inChannels:f,outDepth:R,outHeight:G,outWidth:j,outChannels:J,padInfo:L,strideDepth:S,strideHeight:I,strideWidth:A,filterDepth:d,filterHeight:y,filterWidth:T,effectiveFilterDepth:D,effectiveFilterHeight:O,effectiveFilterWidth:B,dilationDepth:k,dilationHeight:_,dilationWidth:x,inShape:e,outShape:st,filterShape:t}}function ep(e,t,n,s,r){s==null&&(s=Ga(e,t,n));const a=e[0],o=e[1],i=sn((a-t+2*s)/n+1,r),u=sn((o-t+2*s)/n+1,r);return[i,u]}function np(e,t,n,s,r,a){r==null&&(r=Ga(e,t[0],s[0]));const o=[0,0,0,n];for(let i=0;i<3;i++)e[i]+2*r>=t[i]&&(o[i]=sn((e[i]-t[i]+2*r)/s[i]+1,a));return o}function Ga(e,t,n,s=1){const r=Be(t,s);return Math.floor((e[0]*(n-1)-n+r)/2)}function nn(e){return typeof e=="number"?[e,e,e]:e.length===2?[e[0],e[1],1]:e}function gs(e){return typeof e=="number"?[e,e,e]:e}function Be(e,t){return t<=1?e:e+(e-1)*(t-1)}function sp(e,t,n,s,r,a,o,i,u){let c,h,p;if(typeof e=="number"){c={top:e,bottom:e,left:e,right:e,type:e===0?"VALID":"NUMBER"};const d=ep([t,n],a,s,e,i);h=d[0],p=d[1]}else if(e==="same"){h=Math.ceil(t/s),p=Math.ceil(n/r);const f=Math.max(0,(h-1)*s+a-t),d=Math.max(0,(p-1)*r+o-n),y=Math.floor(f/2),T=f-y,N=Math.floor(d/2),S=d-N;c={top:y,bottom:T,left:N,right:S,type:"SAME"}}else if(e==="valid")c={top:0,bottom:0,left:0,right:0,type:"VALID"},h=Math.ceil((t-a+1)/s),p=Math.ceil((n-o+1)/r);else if(typeof e=="object"){const f=u==="channelsLast"?e[1][0]:e[2][0],d=u==="channelsLast"?e[1][1]:e[2][1],y=u==="channelsLast"?e[2][0]:e[3][0],T=u==="channelsLast"?e[2][1]:e[3][1];c={top:f,bottom:d,left:y,right:T,type:f===0&&d===0&&y===0&&T===0?"VALID":"EXPLICIT"},h=sn((t-a+f+d)/s+1,i),p=sn((n-o+y+T)/r+1,i)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:c,outHeight:h,outWidth:p}}function rp(e,t,n,s,r,a,o,i,u,c,h){let p,f,d,y;if(e==="valid"&&(e=0),typeof e=="number"){p={top:e,bottom:e,left:e,right:e,front:e,back:e,type:e===0?"VALID":"NUMBER"};const N=np([t,n,s,1],[i,u,c],1,[r,a,o],e,h);f=N[0],d=N[1],y=N[2]}else if(e==="same"){f=Math.ceil(t/r),d=Math.ceil(n/a),y=Math.ceil(s/o);const T=(f-1)*r+i-t,N=(d-1)*a+u-n,S=(y-1)*o+c-s,I=Math.floor(T/2),A=T-I,k=Math.floor(N/2),_=N-k,x=Math.floor(S/2),D=S-x;p={top:k,bottom:_,left:x,right:D,front:I,back:A,type:"SAME"}}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:p,outDepth:f,outHeight:d,outWidth:y}}function sn(e,t){if(!t)return Math.trunc(e);switch(t){case"round":return Math.round(e);case"ceil":return Math.ceil(e);case"floor":return Math.floor(e);default:throw new Error(`Unknown roundingMode ${t}`)}}function An(e){const[t,n,s]=nn(e);return t===1&&n===1&&s===1}function ae(e,t){return An(e)||An(t)}function We(e){return nn(e).every(t=>t>0)}function ap(e){if(e==="NHWC")return"channelsLast";if(e==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${e}`)}function Ft(e,t,n){if(n!=null){if(typeof t=="string")throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if(typeof t=="number")g(Le(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if(typeof t=="object")t.forEach(s=>{s.forEach(r=>{g(Le(r),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${r}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function op(e,t){const s={x:m(e,"x","reshape","string_or_numeric")},r={shape:t};return w.runKernel(wc,s,r)}const $=b({reshape_:op});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ip(e,t,n,s,r){const a=m(e,"x","avgPool","float32"),o=1;g(ae(n,o),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`);let i=a,u=!1;a.rank===3&&(u=!0,i=$(a,[1,a.shape[0],a.shape[1],a.shape[2]])),g(i.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${i.rank}.`),Ft("avgPool",s,r);const c={x:i},h={filterSize:t,strides:n,pad:s,dimRoundingMode:r};let p=w.runKernel(Ci,c,h);return p=tt(p,a.dtype),u?$(p,[p.shape[1],p.shape[2],p.shape[3]]):p}const Ma=b({avgPool_:ip});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function up(e,t,n,s,r,a="NDHWC"){const o=m(e,"x","avgPool3d","float32");let i=o,u=!1;o.rank===4&&(u=!0,i=$(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),g(i.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${i.rank}.`),g(a==="NDHWC",()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),g(typeof n=="number"&&n>0||Array.isArray(n)&&n[0]>0&&n[1]>0&&n[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${n}'`),Ft("avgPool3d",s,r);const c={x:i},h={filterSize:t,strides:n,pad:s,dimRoundingMode:r,dataFormat:a};let p=w.runKernel(Bi,c,h);return p=tt(p,i.dtype),u?$(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}const cp=b({avgPool3d_:up});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lp(e,t=0){g(e.length>=1,()=>"Pass at least one tensor to concat");const n=en(e,"tensors","concat","string_or_numeric");if(n[0].dtype==="complex64"&&n.forEach(a=>{if(a.dtype!=="complex64")throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${a.dtype}. `)}),n.length===1)return Yt(n[0]);const s=n,r={axis:t};return w.runKernel(Gi,s,r)}const ct=b({concat_:lp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hp(e,t,n=!1,s=!1){let r=m(e,"a","matMul"),a=m(t,"b","matMul");[r,a]=Q(r,a);const o={a:r,b:a},i={transposeA:n,transposeB:s};return w.runKernel(Ri,o,i)}const U=b({matMul_:hp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pp(e){const n={x:m(e,"x","sigmoid","float32")};return w.runKernel(Bc,n)}const Re=b({sigmoid_:pp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fp(e,t,n){const s=m(e,"x","slice","string_or_numeric");if(s.rank===0)throw new Error("Slicing scalar is not possible");const r={x:s},a={begin:t,size:n};return w.runKernel(Dc,r,a)}const q=b({slice_:fp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mp(e){const n={x:m(e,"x","tanh","float32")};return w.runKernel(el,n)}const ys=b({tanh_:mp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dp(e,t,n,s,r,a){const o=m(e,"forgetBias","basicLSTMCell"),i=m(t,"lstmKernel","basicLSTMCell"),u=m(n,"lstmBias","basicLSTMCell"),c=m(s,"data","basicLSTMCell"),h=m(r,"c","basicLSTMCell"),p=m(a,"h","basicLSTMCell"),f=ct([c,p],1),d=U(f,i),y=F(d,u),T=y.shape[0],N=y.shape[1]/4,S=[T,N],I=q(y,[0,0],S),A=q(y,[0,N],S),k=q(y,[0,N*2],S),_=q(y,[0,N*3],S),x=F(v(Re(I),ys(A)),v(h,Re(F(o,k)))),D=v(ys(x),Re(_));return[x,D]}const gp=b({basicLSTMCell_:dp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yp(e,t,n){const s=m(e,"x","batchToSpaceND"),r=t.reduce((i,u)=>i*u);g(s.rank>=1+t.length,()=>`input rank is ${s.rank} but should be > than blockShape.length ${t.length}`),g(n.length===t.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${t.length}`),g(s.shape[0]%r===0,()=>`input tensor batch is ${s.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(" * ")} === ${r}`);const a={x:s},o={blockShape:t,crops:n};return w.runKernel(Pi,a,o)}const Ka=b({batchToSpaceND_:yp});function bp(e){let t;return e.rank===0||e.rank===1?t=$(e,[1,1,1,e.size]):e.rank===2?t=$(e,[1,1,e.shape[0],e.shape[1]]):e.rank===3?t=$(e,[1,e.shape[0],e.shape[1],e.shape[2]]):t=e,t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wp(e,t,n,s,r,a){a==null&&(a=.001);const o=m(e,"x","batchNorm"),i=m(t,"mean","batchNorm"),u=m(n,"variance","batchNorm");let c;r!=null&&(c=m(r,"scale","batchNorm"));let h;s!=null&&(h=m(s,"offset","batchNorm")),g(i.rank===u.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),g(h==null||i.rank===h.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),g(c==null||i.rank===c.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");const f={x:bp(o),scale:c,offset:h,mean:i,variance:u},d={varianceEpsilon:a},y=w.runKernel(Su,f,d);return $(y,o.shape)}const jn=b({batchNorm_:wp});function Np(e,t,n,s,r,a){const o=m(e,"x","batchNorm"),i=m(t,"mean","batchNorm"),u=m(n,"variance","batchNorm");let c;r!=null&&(c=m(r,"scale","batchNorm"));let h;return s!=null&&(h=m(s,"offset","batchNorm")),g(o.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${o.rank}.`),g(i.rank===2||i.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${i.rank}.`),g(u.rank===2||u.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${u.rank}.`),c!=null&&g(c.rank===2||c.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${c.rank}.`),h!=null&&g(h.rank===2||h.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${h.rank}.`),jn(o,i,u,h,c,a)}const Tp=b({batchNorm2d_:Np});function Sp(e,t,n,s,r,a){const o=m(e,"x","batchNorm"),i=m(t,"mean","batchNorm"),u=m(n,"variance","batchNorm");let c;r!=null&&(c=m(r,"scale","batchNorm"));let h;return s!=null&&(h=m(s,"offset","batchNorm")),g(o.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${o.rank}.`),g(i.rank===3||i.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${i.rank}.`),g(u.rank===3||u.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${u.rank}.`),c!=null&&g(c.rank===3||c.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${c.rank}.`),h!=null&&g(h.rank===3||h.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${h.rank}.`),jn(o,i,u,h,c,a)}const $p=b({batchNorm3d_:Sp});function kp(e,t,n,s,r,a){const o=m(e,"x","batchNorm"),i=m(t,"mean","batchNorm"),u=m(n,"variance","batchNorm");let c;r!=null&&(c=m(r,"scale","batchNorm"));let h;return s!=null&&(h=m(s,"offset","batchNorm")),g(o.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${o.rank}.`),g(i.rank===4||i.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${i.rank}.`),g(u.rank===4||u.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${u.rank}.`),c!=null&&g(c.rank===4||c.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${c.rank}.`),h!=null&&g(h.rank===4||h.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${h.rank}.`),jn(o,i,u,h,c,a)}const Ep=b({batchNorm4d_:kp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vp(e,t,n){const s=m(e,"x","bincount"),r=m(t,"weights","bincount");g(s.dtype==="int32",()=>`Error in bincount: input dtype must be int32, but got ${s.dtype}`),g(n>=0,()=>`size must be non-negative, but got ${n}.`),g(r.size===s.size||r.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${s.shape}, weights shape: ${r.shape}.`);const a={x:s,weights:r},o={size:n};return w.runKernel(Li,a,o)}const Ha=b({bincount_:vp});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _p(e,t){const n=m(e,"x","bitwiseAnd"),s=m(t,"y","bitwiseAnd");if(!Pt(n.shape,s.shape))throw new Error(`BitwiseAnd: Tensors must have the same shape. x: ${n.shape}, y: ${s.shape}`);if(n.dtype!=="int32"||s.dtype!=="int32")throw new Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${n.dtype} and type of y: ${s.dtype}`);const r={a:n,b:s};return w.runKernel(zi,r)}const xp=b({bitwiseAnd_:_p});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ip(e,t){const n=m(e,"s0","broadcastArgs","int32"),s=m(t,"s1","broadcastArgs","int32");if(n.rank!==1)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${n.rank}`);if(s.rank!==1)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${s.rank}`);const r={s0:n,s1:s};return w.runKernel(Vi,r)}const Ap=b({broadcastArgs_:Ip});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dp(e,t){let n=m(e,"broadcastTo","x");const s=n.shape;if(bt(t),t.length<n.rank)throw new Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){const c=n.shape.slice();for(;c.length<t.length;)c.unshift(1);n=$(n,c)}const r=n.shape,a=Array.from(t);for(let c=t.length-1;c>=0;c--)if(r[c]===t[c])a[c]=1;else if(n.shape[c]!==1)throw new Error(`broadcastTo(): [${s}] cannot be broadcast to [${t}].`);if(a.map((c,h)=>c>1?h:-1).filter(c=>c>=0).length===0)return Yt(n);const i={x:n},u={reps:a};return w.runKernel(ha,i,u)}const Sn=b({broadcastTo_:Dp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Op(e){const n={x:m(e,"x","ceil","float32")};return w.runKernel(Ui,n)}const Fp=b({ceil_:Op});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dn(e,t,n){bt(e),n=n||hn(t);const s={shape:e,value:t,dtype:n};return w.runKernel(bu,{},s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cp(e,t,n){const s=m(e,"x","clipByValue");if(g(t<=n,()=>`Error in clip: min (${t}) must be less than or equal to max (${n}).`),t===n)return dn(s.shape,t,s.dtype);const r={x:s},a={clipValueMin:t,clipValueMax:n};return w.runKernel(ji,r,a)}const Bp=b({clipByValue_:Cp});function Rp(e){return ct(e,0)}const Pp=b({concat1d_:Rp});function Lp(e,t){return ct(e,t)}const zp=b({concat2d_:Lp});function Vp(e,t){return ct(e,t)}const Up=b({concat3d_:Vp});function jp(e,t){return ct(e,t)}const Wp=b({concat4d_:jp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qp(e,t,n,s,r="NHWC",a=[1,1],o){const i=m(e,"x","conv2d","float32"),u=m(t,"filter","conv2d","float32");let c=i,h=!1;i.rank===3&&(h=!0,c=$(i,[1,i.shape[0],i.shape[1],i.shape[2]])),g(c.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${c.rank}.`),g(u.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${u.rank}.`),Ft("conv2d",s,o);const p=r==="NHWC"?c.shape[3]:c.shape[1];g(p===u.shape[2],()=>`Error in conv2d: depth of input (${p}) must match input depth for filter ${u.shape[2]}.`),g(ae(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),g(We(a),()=>"Error in conv2D: Dilated rates should be larger than 0."),g(We(n),()=>"Error in conv2D: Strides should be larger than 0.");const f={x:c,filter:u},d={strides:n,pad:s,dataFormat:r,dilations:a,dimRoundingMode:o},y=w.runKernel(Mi,f,d);return h?$(y,[y.shape[1],y.shape[2],y.shape[3]]):y}const Wn=b({conv2d_:qp});function Gp(e,t,n,s,r="NWC",a=1,o){const i=m(e,"x","conv1d"),u=m(t,"filter","conv1d");let c=i,h=!1;i.rank===2&&(h=!0,c=$(i,[1,i.shape[0],i.shape[1]])),g(c.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${c.rank}.`),g(u.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${u.rank}.`),Ft("conv1d",s,o),g(c.shape[2]===u.shape[1],()=>`Error in conv1d: depth of input (${c.shape[2]}) must match input depth for filter ${u.shape[1]}.`),g(ae(n,a),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${a}'`),g(We(a),()=>"Error in conv1D: Dilated rates should be larger than 0."),g(We(n),()=>"Error in conv1D: Stride should be larger than 0."),g(r==="NWC",()=>`Error in conv1d: got dataFormat of ${r} but only NWC is currently supported.`);const p=$(u,[1,u.shape[0],u.shape[1],u.shape[2]]),f=$(c,[c.shape[0],1,c.shape[1],c.shape[2]]),N=Wn(f,p,[1,n],s,"NHWC",[1,a],o);return h?$(N,[N.shape[2],N.shape[3]]):$(N,[N.shape[0],N.shape[2],N.shape[3]])}const Mp=b({conv1d_:Gp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kp(e,t,n,s,r,a="NHWC",o){g(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let i=e,u=t,c=!1;t.rank===3&&(c=!0,u=$(t,[1,t.shape[0],t.shape[1],t.shape[2]]),i=[1,e[0],e[1],e[2]]),g(i.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${i.length}.`),g(u.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${u.rank}`),g(n.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);const h=a==="NHWC"?i[3]:i[1],p=a==="NHWC"?u.shape[3]:u.shape[1];g(h===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${h}) must match input depth for filter ${n.shape[2]}.`),g(p===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${p}) must match output depth for filter ${n.shape[3]}.`),Ft("conv2dDerInput",r,o);const f={dy:u,filter:n},d={strides:s,pad:r,dataFormat:a,dimRoundingMode:o,inputShape:i},y=w.runKernel(Hi,f,d);return c?$(y,[y.shape[1],y.shape[2],y.shape[3]]):y}const Xa=b({conv2DBackpropInput_:Kp});function Hp(e,t,n,s,r,a){const o=m(e,"x","conv2dTranspose"),i=m(t,"filter","conv2dTranspose");return Xa(n,o,i,s,r,"NHWC",a)}const Xp=b({conv2dTranspose_:Hp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jp(e,t,n,s,r="NDHWC",a=[1,1,1]){const o=m(e,"x","conv3d"),i=m(t,"filter","conv3d");let u=o,c=!1;o.rank===4&&(c=!0,u=$(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),g(u.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${u.rank}.`),g(i.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${i.rank}.`),g(u.shape[4]===i.shape[3],()=>`Error in conv3d: depth of input (${u.shape[4]}) must match input depth for filter ${i.shape[3]}.`),g(ae(n,a),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),g(r==="NDHWC",()=>`Error in conv3d: got dataFormat of ${r} but only NDHWC is currently supported.`),g(We(a),()=>"Error in conv3D: Dilated rates should be larger than 0."),g(We(n),()=>"Error in conv3D: Strides should be larger than 0.");const h={x:u,filter:i},p={strides:n,pad:s,dataFormat:r,dilations:a},f=w.runKernel(Xi,h,p);return c?$(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}const Yp=b({conv3d_:Jp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zp(e,t,n,s,r){g(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let a=e,o=t,i=!1;t.rank===4&&(i=!0,o=$(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),a=[1,e[0],e[1],e[2],e[3]]);const u=a[4],c=o.shape[4];g(a.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${a.length}.`),g(o.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${o.rank}`),g(n.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),g(u===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${u}) must match input depth for filter ${n.shape[3]}.`),g(c===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${c}) must match output depth for filter ${n.shape[4]}.`);const h={dy:o,filter:n},p={pad:r,strides:s,inputShape:a},f=w.runKernel(Ji,h,p);return i?$(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}const Qp=b({conv3DBackpropInput_:Zp});function tf(e,t,n,s,r){const a=m(e,"x","conv3dTranspose"),o=m(t,"filter","conv3dTranspose");return Qp(n,a,o,s,r)}const ef=b({conv3dTranspose_:tf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nf(e){const n={x:m(e,"x","cos","float32")};return w.runKernel(Yi,n)}const sf=b({cos_:nf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rf(e){const n={x:m(e,"x","cosh","float32")};return w.runKernel(Zi,n)}const af=b({cosh_:rf});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function of(e,t=0,n=!1,s=!1){const a={x:m(e,"x","cumprod")},o={axis:t,exclusive:n,reverse:s};return w.runKernel(Qi,a,o)}const uf=b({cumprod_:of});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cf(e,t=0,n=!1,s=!1){const a={x:m(e,"x","cumsum")},o={axis:t,exclusive:n,reverse:s};return w.runKernel(tu,a,o)}const lf=b({cumsum_:cf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hf(e,t,n,s=!1){const r=m(e,"x","denseBincount"),a=m(t,"weights","denseBincount");g(r.dtype==="int32",()=>`Error in denseBincount: input dtype must be int32, but got ${r.dtype}`),g(r.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${r.rank}.`),g(n>=0,()=>`size must be non-negative, but got ${n}.`),g(a.size===r.size||a.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${r.shape}, weights shape: ${a.shape}.`);const o={x:r,weights:a},i={size:n,binaryOutput:s};return w.runKernel(nu,o,i)}const pf=b({denseBincount_:hf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ff(e,t,n="NHWC"){const s=m(e,"x","depthToSpace","float32"),r=n==="NHWC"?s.shape[1]:s.shape[2],a=n==="NHWC"?s.shape[2]:s.shape[3],o=n==="NHWC"?s.shape[3]:s.shape[1];g(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),g(r*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${r} and ${t}  for depthToSpace with input shape
    ${s.shape}`),g(a*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${a} and ${t} for depthToSpace with input shape
        ${s.shape}`),g(o%(t*t)===0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${o} for depthToSpace with input shape ${s.shape}`);const i={x:s},u={blockSize:t,dataFormat:n};return w.runKernel(su,i,u)}const mf=b({depthToSpace_:ff});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function df(e,t,n,s,r="NHWC",a=[1,1],o){const i=m(e,"x","depthwiseConv2d","float32"),u=m(t,"filter","depthwiseConv2d","float32");let c=i,h=!1;i.rank===3&&(h=!0,c=$(i,[1,i.shape[0],i.shape[1],i.shape[2]])),g(c.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${c.rank}.`),g(u.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${u.rank}.`);const p=r==="NHWC"?c.shape[3]:c.shape[1];g(p===u.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${p}) must match the inChannels dimension in filter ${u.shape[2]}.`),Ft("depthwiseConv2d",s,o);const f={x:c,filter:u},d={strides:n,pad:s,dataFormat:r,dilations:a,dimRoundingMode:o},y=w.runKernel(ru,f,d);return h?$(y,[y.shape[1],y.shape[2],y.shape[3]]):y}const qs=b({depthwiseConv2d_:df});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gf(e){const n={x:m(e,"x","diag")};return w.runKernel(iu,n)}const yf=b({diag_:gf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bf(e,t,n,s,r=[1,1],a="NHWC"){const o=m(e,"x","dilation2d"),i=m(t,"filter","dilation2d");g(o.rank===3||o.rank===4,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${o.rank}.`),g(i.rank===3,()=>`Error in dilation2d: filter must be rank 3, but got rank ${i.rank}.`),g(a==="NHWC",()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${a}`);let u=o,c=!1;o.rank===3&&(u=$(o,[1,o.shape[0],o.shape[1],o.shape[2]]),c=!0),g(u.shape[3]===i.shape[2],()=>`Error in dilation2d:  input and filter must have the same depth: ${u.shape[3]} vs ${i.shape[2]}`);const h={x:u,filter:i},p={strides:n,pad:s,dilations:r},f=w.runKernel(uu,h,p);return c?$(f,[f.shape[1],f.shape[2],f.shape[3]]):f}const wf=b({dilation2d_:bf});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nf(e,t){const n=e.length,s=[];for(let r=0;r<n;r++){const a=n-1-r,o=e[a]||1;(t[t.length-1-r]||1)>1&&o===1&&s.unshift(a)}return s}function Ja(e,t){const n=[];for(let s=0;s<t.length;s++){const r=e[e.length-s-1],a=t.length-s-1,o=t[a];(r==null||r===1&&o>1)&&n.unshift(a)}return n}function nt(e,t){const n=Math.max(e.length,t.length),s=new Array(n);for(let r=0;r<n;r++){let a=e[e.length-r-1];a==null&&(a=1);let o=t[t.length-r-1];if(o==null&&(o=1),a===1)s[n-r-1]=o;else if(o===1)s[n-r-1]=a;else if(a!==o){const i=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(i)}else s[n-r-1]=a}return s}const BN=Object.freeze(Object.defineProperty({__proto__:null,assertAndGetBroadcastShape:nt,getBroadcastDims:Nf,getReductionAxes:Ja},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tf(e,t){let n=m(e,"a","equal","string_or_numeric"),s=m(t,"b","equal","string_or_numeric");[n,s]=Q(n,s),nt(n.shape,s.shape);const r={a:n,b:s};return w.runKernel(fu,r)}const Ya=b({equal_:Tf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sf(e,t,n){const s=m(t,"a","where"),r=m(n,"b","where"),a=m(e,"condition","where","bool"),o=nt(nt(a.shape,s.shape),r.shape),i=Sn(a,o),u=Sn(s,o),c=Sn(r,o),h={condition:i,t:u,e:c};return w.runKernel(Ic,h)}const Zt=b({where_:Sf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $f(e){const n={x:m(e,"x","zerosLike")};return w.runKernel(il,n)}const Tt=b({zerosLike_:$f});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kf(e,t){let n=m(e,"a","div"),s=m(t,"b","div");[n,s]=Q(n,s);const r=K(n,s),a=Tt(r),o=Ya(s,a);return Zt(o,a,r)}const Ef=b({divNoNan_:kf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vf(e,t){const n=m(e,"t1","dot"),s=m(t,"t2","dot");g((n.rank===1||n.rank===2)&&(s.rank===1||s.rank===2),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${s.rank}.`);const r=n.rank===1?n.size:n.shape[1],a=s.rank===1?s.size:s.shape[0];if(g(r===a,()=>`Error in dot: inner dimensions of inputs must match, but got ${r} and ${a}.`),n.rank===1&&s.rank===1){const o=$(n,[1,-1]),i=$(s,[-1,1]),u=U(o,i);return $(u,[])}else if(n.rank===1&&s.rank===2){const o=$(n,[1,-1]),i=$(s,[s.shape[0],s.shape[1]]),u=U(o,i);return $(u,[u.size])}else if(n.rank===2&&s.rank===1){const o=$(s,[-1,1]),i=U(n,o);return $(i,[i.size])}else{const o=$(s,[s.shape[0],s.shape[1]]);return U(n,o)}}const _f=b({dot_:vf});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xf(e,...t){const n=t.map((r,a)=>m(r,`tensors${a}`,"einsum")),s={equation:e};return w.runKernel(lu,n,s)}const Oe=b({einsum_:xf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function If(e){const n={x:m(e,"x","elu","float32")};return w.runKernel(hu,n)}const Za=b({elu_:If});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Af(e,t){const n=m(e,"x","ensureShape","string_or_numeric");if(!Jr(n.shape,t))throw new Error(`EnsureShape: Shape of tensor ${n.shape} is not compatible with expected shape ${t}`);return e}const Df=b({ensureShape_:Af});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Of(e){let t=m(e,"x","erf");g(t.dtype==="int32"||t.dtype==="float32",()=>"Input dtype must be `int32` or `float32`."),t.dtype==="int32"&&(t=tt(t,"float32"));const n={x:t};return w.runKernel(pu,n)}const Ff=b({erf_:Of});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qa(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function Cf(e,t,n){const s=e.length+t.length,r=[];let a=0,o=0;for(let i=0;i<s;i++)n.indexOf(i)===-1?r.push(e[a++]):r.push(t[o++]);return r}function RN(e,t){const n=[],s=e.length;for(let a=0;a<s;a++)t.indexOf(a)===-1&&n.push(e[a]);const r=t.map(a=>e[a]);return[n,r]}function qn(e,t){const n=t.map(s=>1);return Cf(e,n,t)}function PN(e,t,n){g(Qa(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function LN(e,t){if(Qa(e,t))return null;const n=[];for(let s=0;s<t;++s)e.indexOf(s)===-1&&n.push(s);return e.forEach(s=>n.push(s)),n}function zN(e){return e.map((t,n)=>[n,t]).sort((t,n)=>t[1]-n[1]).map(t=>t[0])}function VN(e,t){const n=[];for(let s=t-e;s<t;++s)n.push(s);return n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bf(e,t=null,n=!1){const r={x:m(e,"x","max")},a={reductionIndices:t,keepDims:n};return w.runKernel(ju,r,a)}const Pe=b({max_:Bf});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rf(e,t=null,n=!1){const r={x:m(e,"x","min")},a={axis:t,keepDims:n};return w.runKernel(Hu,r,a)}const bs=b({min_:Rf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pf(e,t){let n=m(e,"base","pow"),s=m(t,"exp","pow");[n,s]=Q(n,s);const r={a:n,b:s};return w.runKernel(cc,r)}const rn=b({pow_:Pf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function z(e,t){if((St(e)&&t!=="string"||Array.isArray(e))&&t!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(t==="string"&&St(e)&&!(e instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return re(e,[],[],t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lf(e){const n={x:m(e,"x","sqrt","float32")};return w.runKernel(Pc,n)}const Wt=b({sqrt_:Lf});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zf(e){const t=m(e,"x","square"),n={};return w.runKernel("Square",{x:t},n)}const At=b({square_:zf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vf(e,t=null,n=!1){let s=m(e,"x","sum");s.dtype==="bool"&&(s=tt(s,"int32"));const r={x:s},a={axis:t,keepDims:n};return w.runKernel(Lc,r,a)}const M=b({sum_:Vf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uf(e,t="euclidean",n=null,s=!1){e=m(e,"x","norm");const r=to(e,t,n);let a=r.shape;if(s){const o=ln(n,e.shape);a=qn(r.shape,o)}return $(r,a)}function to(e,t,n=null){if(e.rank===0)return Nt(e);if(e.rank!==1&&n===null)return to($(e,[-1]),t,n);if(e.rank===1||typeof n=="number"||Array.isArray(n)&&n.length===1){if(t===1)return M(Nt(e),n);if(t===1/0)return Pe(Nt(e),n);if(t===-1/0)return bs(Nt(e),n);if(t==="euclidean"||t===2)return Wt(M(rn(Nt(e),z(2,"int32")),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(n)&&n.length===2){if(t===1)return Pe(M(Nt(e),n[0]),n[1]-1);if(t===1/0)return Pe(M(Nt(e),n[1]),n[0]);if(t===-1/0)return bs(M(Nt(e),n[1]),n[0]);if(t==="fro"||t==="euclidean")return Wt(M(At(e),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}throw new Error(`Error in norm: invalid axis: ${n}`)}const Gn=b({norm_:Uf});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jf(e,t=null,n=!1){return Gn(e,"euclidean",t,n)}const Wf=b({euclideanNorm_:jf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qf(e){const n={x:m(e,"x","exp")};return w.runKernel(mu,n)}const be=b({exp_:qf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gf(e,t=0){const n=m(e,"x","expandDims","string_or_numeric");g(t<=n.rank,()=>"Axis must be <= rank of the tensor");const s={input:n},r={dim:t};return w.runKernel(du,s,r)}const Kt=b({expandDims_:Gf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mf(e){const n={x:m(e,"x","expm1")};return w.runKernel(gu,n)}const Kf=b({expm1_:Mf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hf(e,t){const n=m(e,"x","tile","string_or_numeric");g(n.rank===t.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${t}.`);const s={x:n},r={reps:t};return w.runKernel(ha,s,r)}const Ye=b({tile_:Hf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xf(e,t,n,s="float32"){t==null&&(t=e);const r=jt([e,t],s),a=e<=t?e:t;for(let i=0;i<a;++i)r.set(1,i,i);const o=$(r.toTensor(),[e,t]);if(n==null)return o;if(n.length===1)return Ye(Kt(o,0),[n[0],1,1]);if(n.length===2)return Ye(Kt(Kt(o,0),0),[n[0],n[1],1,1]);if(n.length===3)return Ye(Kt(Kt(Kt(o,0),0),0),[n[0],n[1],n[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}const eo=b({eye_:Xf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jf(e){const n={x:m(e,"x","floor","float32")};return w.runKernel(Nu,n)}const no=b({floor_:Jf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yf(e,t,n=0,s=0){const r=m(e,"x","gather"),a=m(t,"indices","gather","int32"),o={x:r,indices:a},i={axis:n,batchDims:s};return w.runKernel($u,o,i)}const so=b({gather_:Yf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zf(e,t){let n=m(e,"a","greater","string_or_numeric"),s=m(t,"b","greater","string_or_numeric");[n,s]=Q(n,s),nt(n.shape,s.shape);const r={a:n,b:s};return w.runKernel(Eu,r)}const Mn=b({greater_:Zf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qf(e,t){let n=m(e,"a","greaterEqual","string_or_numeric"),s=m(t,"b","greaterEqual","string_or_numeric");[n,s]=Q(n,s),nt(n.shape,s.shape);const r={a:n,b:s};return w.runKernel(vu,r)}const ro=b({greaterEqual_:Qf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tm(e){const n={input:m(e,"input","imag")};return w.runKernel(xu,n)}const Kn=b({imag_:tm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function em(e){const n={x:m(e,"x","isFinite")};return w.runKernel(Iu,n)}const nm=b({isFinite_:em});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sm(e){const n={x:m(e,"x","isInf")};return w.runKernel(Au,n)}const rm=b({isInf_:sm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function am(e){const n={x:m(e,"x","isNaN")};return w.runKernel(Du,n)}const om=b({isNaN_:am});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function im(e,t=.2){const s={x:m(e,"x","leakyRelu")},r={alpha:t};return w.runKernel(Ou,s,r)}const ao=b({leakyRelu_:im});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function um(e,t){let n=m(e,"a","less","string_or_numeric"),s=m(t,"b","less","string_or_numeric");[n,s]=Q(n,s),nt(n.shape,s.shape);const r={a:n,b:s};return w.runKernel(Fu,r)}const ws=b({less_:um});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cm(e,t){let n=m(e,"a","lessEqual","string_or_numeric"),s=m(t,"b","lessEqual","string_or_numeric");[n,s]=Q(n,s),nt(n.shape,s.shape);const r={a:n,b:s};return w.runKernel(Cu,r)}const Gs=b({lessEqual_:cm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lm(e,t,n){if(n<=0)throw new Error("The number of values should be positive.");const s={start:e,stop:t,num:n};return w.runKernel(Bu,{},s)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hm(e,t=5,n=1,s=1,r=.5){const a=m(e,"x","localResponseNormalization");g(a.rank===4||a.rank===3,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${a.rank}.`),g(Le(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let o=a,i=!1;a.rank===3&&(i=!0,o=$(a,[1,a.shape[0],a.shape[1],a.shape[2]]));const u={x:o},c={depthRadius:t,bias:n,alpha:s,beta:r},h=w.runKernel(Uu,u,c);return i?$(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const pm=b({localResponseNormalization_:hm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fm(e){const n={x:m(e,"x","log","float32")};return w.runKernel(Ru,n)}const an=b({log_:fm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mm(e){const n={x:m(e,"x","log1p")};return w.runKernel(Pu,n)}const oo=b({log1p_:mm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function UN(e){return g(Qt(e),()=>"The f passed in grad(f) must be a function"),(t,n)=>{const s=m(t,"x","tf.grad","string_or_numeric"),r=n!=null?m(n,"dy","tf.grad"):null;return w.tidy(()=>{const{value:a,grads:o}=w.gradients(()=>e(s),[s],r);return r!=null&&lt(a.shape,r.shape,"The shape of dy passed in grad(f)(x, dy) must match the shape returned by f(x)"),Hn(o),o[0]})}}function jN(e){return g(Qt(e),()=>"The f passed in grads(f) must be a function"),(t,n)=>{g(Array.isArray(t),()=>"The args passed in grads(f)(args) must be an array of `Tensor`s or `TensorLike`s");const s=en(t,"args","tf.grads","string_or_numeric"),r=n!=null?m(n,"dy","tf.grads"):null;return w.tidy(()=>{const{value:a,grads:o}=w.gradients(()=>e(...s),s,r);return r!=null&&lt(a.shape,r.shape,"The shape of dy passed in grads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),Hn(o),o})}}function WN(e){return g(Qt(e),()=>"The f passed in valueAndGrad(f) must be a function"),(t,n)=>{g(t instanceof et,()=>"The x passed in valueAndGrad(f)(x) must be a tensor"),g(n==null||n instanceof et,()=>"The dy passed in valueAndGrad(f)(x, dy) must be a tensor");const{grads:s,value:r}=w.gradients(()=>e(t),[t],n);return Hn(s),{grad:s[0],value:r}}}function qN(e){return g(Qt(e),()=>"The f passed in valueAndGrads(f) must be a function"),(t,n)=>{g(Array.isArray(t)&&t.every(r=>r instanceof et),()=>"The args passed in valueAndGrads(f)(args) must be array of tensors"),g(n==null||n instanceof et,()=>"The dy passed in valueAndGrads(f)(args, dy) must be a tensor");const s=w.gradients(()=>e(...t),t,n);return n!=null&&lt(s.value.shape,n.shape,"The shape of dy passed in valueAndGrads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),Hn(s.grads),s}}function dm(e,t){g(Qt(e),()=>"The f passed in variableGrads(f) must be a function"),g(t==null||Array.isArray(t)&&t.every(c=>c instanceof xn),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const n=t!=null;if(!n){t=[];for(const c in w.registeredVariables)t.push(w.registeredVariables[c])}const s=n?t.filter(c=>!c.trainable):null,r=t.length;t=t.filter(c=>c.trainable),g(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${r} variables is trainable.`);const a=!0,{value:o,grads:i}=w.gradients(e,t,null,a);g(i.some(c=>c!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),g(o.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${o.rank} tensor`);const u={};return t.forEach((c,h)=>{i[h]!=null&&(u[c.name]=i[h])}),s!=null&&s.forEach(c=>u[c.name]=null),{value:o,grads:u}}function qt(e){return w.customGrad(e)}function Hn(e){if(e.filter(n=>n==null).length>0)throw new Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that
    the f you passed encloses all operations that lead from x to y.`)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gm(e){const n={x:m(e,"x","neg")};return w.runKernel(tc,n)}const Rt=b({neg_:gm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ym(e){const n={x:m(e,"x","softplus")};return w.runKernel(Rc,n)}const io=b({softplus_:ym});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bm(e){const t=m(e,"x","logSigmoid");return qt(s=>({value:Rt(io(Rt(s))),gradFunc:o=>v(o,Re(Rt(s)))}))(t)}const wm=b({logSigmoid_:bm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nm(e,t){let n=m(e,"a","sub"),s=m(t,"b","sub");[n,s]=Q(n,s);const r={a:n,b:s};return w.runKernel(Qc,r)}const P=b({sub_:Nm});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tm(e,t=-1){const n=m(e,"logits","logSoftmax");if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${t}`);return qt((r,a)=>{const i=Pe(r,t,!0),u=P(r,i),c=P(tt(u,"float32"),an(M(be(u),t,!0)));return a([c]),{value:c,gradFunc:(p,f)=>{const[d]=f,y=!0,T=be(d);return P(p,v(M(p,t,y),T))}}})(n)}const Sm=b({logSoftmax_:Tm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $m(e,t=null,n=!1){const s=m(e,"x","logSumExp"),r=ln(t,s.shape),a=Pe(s,r,!0),o=P(s,a),i=be(o),u=M(i,r),c=an(u),h=F($(a,c.shape),c);if(n){const p=qn(h.shape,r);return $(h,p)}return h}const uo=b({logSumExp_:$m});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function km(e,t){const n=m(e,"a","logicalAnd","bool"),s=m(t,"b","logicalAnd","bool");nt(n.shape,s.shape);const r={a:n,b:s};return w.runKernel(Lu,r)}const Dn=b({logicalAnd_:km});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Em(e){const n={x:m(e,"x","logicalNot","bool")};return w.runKernel(zu,n)}const co=b({logicalNot_:Em});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vm(e,t){const n=m(e,"a","logicalOr","bool"),s=m(t,"b","logicalOr","bool");nt(n.shape,s.shape);const r={a:n,b:s};return w.runKernel(Vu,r)}const lo=b({logicalOr_:vm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _m(e,t){const n=m(e,"a","logicalXor","bool"),s=m(t,"b","logicalXor","bool");return nt(n.shape,s.shape),Dn(lo(e,t),co(Dn(e,t)))}const xm=b({logicalXor_:_m});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bn=2147483648;function Im(e,t,n="left"){const s=m(e,"sortedSequence","searchSorted"),r=m(t,"values","searchSorted"),a=s.shape[s.shape.length-1],o=r.shape[r.shape.length-1],i=$(s,[-1,a]),u=$(r,[-1,o]);if(i.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(i.shape[0]!==u.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(Z(u.shape)>=bn)throw new Error(`values tensor size must less than ${bn}`);if(i.shape[1]>=bn)throw new Error(`trailing dim_size must less than ${bn} for int32 output type, was ${i.shape[1]}`);const c={sortedSequence:i,values:u},h={side:n};return w.runKernel(xc,c,h)}const Ms=b({searchSorted_:Im});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Am(e,t){return Ms(e,t,"left")}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dm(e,t,n,s,r){const a=m(e,"x","maxPool"),o=1;let i=a,u=!1;a.rank===3&&(u=!0,i=$(a,[1,a.shape[0],a.shape[1],a.shape[2]])),g(i.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${i.rank}.`),g(ae(n,o),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`),Ft("maxPool",s,r);const c={x:i},h={filterSize:t,strides:n,pad:s,dimRoundingMode:r},p=w.runKernel(qu,c,h);return u?$(p,[p.shape[1],p.shape[2],p.shape[3]]):p}const ho=b({maxPool_:Dm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Om(e,t=[1,1,1],n,s,r,a="NDHWC"){const o=m(e,"x","maxPool3d");let i=o,u=!1;o.rank===4&&(u=!0,i=$(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),g(i.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${i.rank}.`),g(a==="NDHWC",()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),Ft("maxPool3d",s,r);const c={x:i},h={filterSize:t,strides:n,pad:s,dimRoundingMode:r,dataFormat:a},p=w.runKernel(Gu,c,h);return u?$(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}const Fm=b({maxPool3d_:Om});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cm(e,t,n,s,r=!1){const o={x:m(e,"x","maxPoolWithArgmax")},i={filterSize:t,strides:n,pad:s,includeBatchInIndex:r},u=w.runKernel(Mu,o,i);return{result:u[0],indexes:u[1]}}const Bm=b({maxPoolWithArgmax_:Cm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rm(e,t){let n=m(e,"a","maximum"),s=m(t,"b","maximum");[n,s]=Q(n,s),n.dtype==="bool"&&(n=tt(n,"int32"),s=tt(s,"int32")),nt(n.shape,s.shape);const r={a:n,b:s};return w.runKernel(Wu,r)}const po=b({maximum_:Rm});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pm(e,t=null,n=!1){const r={x:m(e,"x","mean")},a={axis:t,keepDims:n};return w.runKernel(Ku,r,a)}const On=b({mean_:Pm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qe(e,t="float32"){if(bt(e),t==="complex64"){const s=qe(e,"float32"),r=qe(e,"float32");return ee(s,r)}const n=zn(Z(e),t);return w.makeTensor(n,e,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fe(e,t="float32"){if(bt(e),t==="complex64"){const s=fe(e,"float32"),r=qe(e,"float32");return ee(s,r)}const n=Bs(Z(e),t);return w.makeTensor(n,e,t)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lm(e,t,{indexing:n="xy"}={}){if(n!=="xy"&&n!=="ij")throw new TypeError(`${n} is not a valid third argument to meshgrid`);if(e===void 0)return[];let s=m(e,"x","meshgrid",e instanceof et?e.dtype:"float32");if(t===void 0)return[s];let r=m(t,"y","meshgrid",t instanceof et?t.dtype:"float32");const a=Z(s.shape),o=Z(r.shape);return n==="xy"?(s=$(s,[1,-1]),r=$(r,[-1,1]),[U(fe([o,1],s.dtype),s),U(r,fe([1,a],r.dtype))]):(s=$(s,[-1,1]),r=$(r,[1,-1]),[U(s,fe([1,o],s.dtype)),U(fe([a,1],r.dtype),r)])}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zm(e,t){let n=m(e,"a","minimum"),s=m(t,"b","minimum");[n,s]=Q(n,s),n.dtype==="bool"&&(n=tt(n,"int32"),s=tt(s,"int32")),nt(n.shape,s.shape);const r={a:n,b:s};return w.runKernel(Xu,r)}const Fn=b({minimum_:zm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vm(e,t,n){g(n==="reflect"||n==="symmetric",()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${n}.`);const s=m(e,"x","mirrorPad");if(s.rank===0)throw new Error("mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad");g(t.length===s.rank,()=>`Padding doesn't match input. Must be ${s.rank}. Got ${t.length}.`);const r=n==="reflect"?1:0;for(let i=0;i<s.rank;i++)g(t[i].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),g(t[i][0]>=0&&t[i][0]<=s.shape[i]-r&&t[i][1]>=0&&t[i][1]<=s.shape[i]-r,()=>`Padding in dimension ${i} cannot be greater than or equal to ${s.shape[i]-r} or less than 0 for input of shape ${s.shape}`);const a={paddings:t,mode:n},o={x:s};return w.runKernel(Ju,o,a)}const Um=b({mirrorPad_:Vm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jm(e,t){let n=m(e,"a","mod"),s=m(t,"b","mod");[n,s]=Q(n,s);const r={a:n,b:s};return w.runKernel(Yu,r)}const Wm=b({mod_:jm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qm(e,t=null,n=!1){e=m(e,"x","moments");const s=ln(t,e.shape),r=On(e,s,n);let a=r.shape;n||(a=qn(r.shape,s));const o=At(P(tt(e,"float32"),$(r,a))),i=On(o,s,n);return{mean:r,variance:i}}const Gm=b({moments_:qm});function Mm(e,t,n,s){const r=m(t,"data","multiRNNCell"),a=en(n,"c","multiRNNCell"),o=en(s,"h","multiRNNCell");let i=r;const u=[];for(let p=0;p<e.length;p++){const f=e[p](i,a[p],o[p]);u.push(f[0]),u.push(f[1]),i=f[1]}const c=[],h=[];for(let p=0;p<u.length;p+=2)c.push(u[p]),h.push(u[p+1]);return[c,h]}const Km=b({multiRNNCell_:Mm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hm(e,t,n,s=!1){const r=m(e,"logits","multinomial"),a=r.size,o=r.rank;if(a<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${a}.`);if(o>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${o}`);n=n||Math.random();const u={logits:o===1?$(r,[1,-1]):r},c={numSamples:t,seed:n,normalized:s},h=w.runKernel(Zu,u,c);return o===1?$(h,[h.size]):h}const Xm=b({multinomial_:Hm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jm(e,t){let n=m(e,"a","notEqual","string_or_numeric"),s=m(t,"b","notEqual","string_or_numeric");[n,s]=Q(n,s),nt(n.shape,s.shape);const r={a:n,b:s};return w.runKernel(ec,r)}const fo=b({notEqual_:Jm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ym(e,t,n=1,s=0,r="int32"){if(t<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${t}`);const o={indices:m(e,"indices","oneHot","int32")},i={dtype:r,depth:t,onValue:n,offValue:s};return w.runKernel(oc,o,i)}const Zm=b({oneHot_:Ym});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qm(e){const n={x:m(e,"x","onesLike")};return w.runKernel(ac,n)}const td=b({onesLike_:Qm});function ed(e,t){const n=m(e,"v1","outerProduct"),s=m(t,"v2","outerProduct");g(n.rank===1&&s.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${n.rank} and ${s.rank}.`);const r=$(n,[-1,1]),a=$(s,[1,-1]);return U(r,a)}const nd=b({outerProduct_:ed});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sd(e,t,n=0){const s=m(e,"x","pad");if(s.rank===0)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const r={paddings:t,constantValue:n},a={x:s};return w.runKernel(uc,a,r)}const gn=b({pad_:sd});function rd(e,t,n=0){return g(t.length===2,()=>"Invalid number of paddings. Must be length of 2."),gn(e,[t],n)}const ad=b({pad1d_:rd});function od(e,t,n=0){return g(t.length===2&&t[0].length===2&&t[1].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),gn(e,t,n)}const id=b({pad2d_:od});function ud(e,t,n=0){return g(t.length===3&&t[0].length===2&&t[1].length===2&&t[2].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),gn(e,t,n)}const cd=b({pad3d_:ud});function ld(e,t,n=0){return g(t.length===4&&t[0].length===2&&t[1].length===2&&t[2].length===2&&t[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),gn(e,t,n)}const hd=b({pad4d_:ld});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pd(e,t,n){const s=m(e,"x","spaceToBatchND");g(s.rank>=1+t.length,()=>`input rank ${s.rank} should be > than [blockShape] ${t.length}`),g(n.length===t.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${t.length}`),g(s.shape.reduce((o,i,u)=>u>0&&u<=t.length?o&&(i+n[u-1][0]+n[u-1][1])%t[u-1]===0:o,!0),()=>`input spatial dimensions ${s.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${t.toString()}`);const r={x:s},a={blockShape:t,paddings:n};return w.runKernel(zc,r,a)}const mo=b({spaceToBatchND_:pd});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fd(e,t,n,s,r,a,o){r==null&&(r=[1,1]),a==null&&(a=1),s===0&&(s="valid");const i=m(e,"x","maxPool");let u=i,c=!1;i.rank===3&&(c=!0,u=$(i,[1,i.shape[0],i.shape[1],i.shape[2]])),g(ae(a,r),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${a} and dilations '${r}'`);const h=Qh(u.shape,t,a,r,s),p=[h.dilationHeight,h.dilationWidth];let f;s==="same"?f=dd([h.filterHeight,h.filterWidth],p):f=[[0,0],[0,0]];const d=p[0]===1&&p[1]===1,[y,T]=md([h.inHeight,h.inWidth],p,f),N=d?s:"valid",S=d?u:mo(u,p,y),A=(n==="avg"?()=>Ma(S,t,a,N,o):()=>ho(S,t,a,N,o))(),k=d?A:Ka(A,p,T);return c?$(k,[k.shape[1],k.shape[2],k.shape[3]]):k}function md(e,t,n){const s=n.map(h=>h[0]),r=n.map(h=>h[1]),a=e.concat(s,r),o=t.map((h,p)=>(h-a[p]%h)%h),i=r.map((h,p)=>h+o[p]),u=t.map((h,p)=>[s[p],i[p]]),c=t.map((h,p)=>[0,o[p]]);return[u,c]}function dd(e,t){const s=e.map((o,i)=>o+(o-1)*(t[i]-1)).map(o=>o-1),r=s.map(o=>Math.floor(o/2)),a=s.map((o,i)=>o-r[i]);return s.map((o,i)=>[r[i],a[i]])}const gd=b({pool_:fd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yd(e,t){const n=m(e,"x","prelu"),s=m(t,"alpha","prelu"),r={x:n,alpha:s};return w.runKernel(lc,r)}const go=b({prelu_:yd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bd(e,t=null,n=!1){let s=m(e,"x","prod");s.dtype==="bool"&&(s=tt(s,"int32"));const r={x:s},a={axis:t,keepDims:n};return w.runKernel(hc,r,a)}const wd=b({prod_:bd});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nd(e,t,n,s){const r=e.map((h,p)=>m(h,`tensors${p}`,"raggedGather","int32")),a=m(t,"paramsDenseValues","raggedGather"),o=m(n,"indices","raggedGather","int32"),i={paramsNestedSplits:r,paramsDenseValues:a,indices:o},u={outputRaggedRank:s},c=w.runKernel(pc,i,u);return{outputNestedSplits:c.slice(0,c.length-1),outputDenseValues:c[c.length-1]}}const Td=b({raggedGather_:Nd});/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sd(e,t,n){const s=m(e,"starts","raggedRange"),r=m(t,"limits","raggedRange",s.dtype),a=m(n,"deltas","raggedRange",s.dtype),o={starts:s,limits:r,deltas:a},i=w.runKernel(fc,o);return{rtNestedSplits:i[0],rtDenseValues:i[1]}}const $d=b({raggedRange_:Sd});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kd(e,t,n,s,r){const a=m(e,"shape","raggedTensorToTensor","int32"),o=m(t,"values","raggedTensorToTensor"),i=m(n,"defaultValue","raggedTensorToTensor",o.dtype),u=s.map((p,f)=>m(p,`tensors${f}`,"raggedTensorToTensor","int32")),c={shape:a,values:o,defaultValue:i,rowPartitionTensors:u},h={rowPartitionTypes:r};return w.runKernel(mc,c,h)}const Ed=b({raggedTensorToTensor_:kd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vd(e,t,n){bt(e);const s=Z(e);let r=null;if(n==null||n==="float32")r=new Float32Array(s);else if(n==="int32")r=new Int32Array(s);else if(n==="bool")r=new Uint8Array(s);else throw new Error(`Unknown data type ${n}`);for(let a=0;a<s;a++)r[a]=t();return w.makeTensor(r,e,n)}const _d=b({rand_:vd});var Ks={exports:{}};Ks.exports;(function(e){(function(t,n,s){function r(u){var c=this,h=i();c.next=function(){var p=2091639*c.s0+c.c*23283064365386963e-26;return c.s0=c.s1,c.s1=c.s2,c.s2=p-(c.c=p|0)},c.c=1,c.s0=h(" "),c.s1=h(" "),c.s2=h(" "),c.s0-=h(u),c.s0<0&&(c.s0+=1),c.s1-=h(u),c.s1<0&&(c.s1+=1),c.s2-=h(u),c.s2<0&&(c.s2+=1),h=null}function a(u,c){return c.c=u.c,c.s0=u.s0,c.s1=u.s1,c.s2=u.s2,c}function o(u,c){var h=new r(u),p=c&&c.state,f=h.next;return f.int32=function(){return h.next()*4294967296|0},f.double=function(){return f()+(f()*2097152|0)*11102230246251565e-32},f.quick=f,p&&(typeof p=="object"&&a(p,h),f.state=function(){return a(h,{})}),f}function i(){var u=4022871197,c=function(h){h=String(h);for(var p=0;p<h.length;p++){u+=h.charCodeAt(p);var f=.02519603282416938*u;u=f>>>0,f-=u,f*=u,u=f>>>0,f-=u,u+=f*4294967296}return(u>>>0)*23283064365386963e-26};return c}n&&n.exports?n.exports=o:this.alea=o})(Se,e)})(Ks);var xd=Ks.exports,Hs={exports:{}};Hs.exports;(function(e){(function(t,n,s){function r(i){var u=this,c="";u.x=0,u.y=0,u.z=0,u.w=0,u.next=function(){var p=u.x^u.x<<11;return u.x=u.y,u.y=u.z,u.z=u.w,u.w^=u.w>>>19^p^p>>>8},i===(i|0)?u.x=i:c+=i;for(var h=0;h<c.length+64;h++)u.x^=c.charCodeAt(h)|0,u.next()}function a(i,u){return u.x=i.x,u.y=i.y,u.z=i.z,u.w=i.w,u}function o(i,u){var c=new r(i),h=u&&u.state,p=function(){return(c.next()>>>0)/4294967296};return p.double=function(){do var f=c.next()>>>11,d=(c.next()>>>0)/4294967296,y=(f+d)/(1<<21);while(y===0);return y},p.int32=c.next,p.quick=p,h&&(typeof h=="object"&&a(h,c),p.state=function(){return a(c,{})}),p}n&&n.exports?n.exports=o:this.xor128=o})(Se,e)})(Hs);var Id=Hs.exports,Xs={exports:{}};Xs.exports;(function(e){(function(t,n,s){function r(i){var u=this,c="";u.next=function(){var p=u.x^u.x>>>2;return u.x=u.y,u.y=u.z,u.z=u.w,u.w=u.v,(u.d=u.d+362437|0)+(u.v=u.v^u.v<<4^(p^p<<1))|0},u.x=0,u.y=0,u.z=0,u.w=0,u.v=0,i===(i|0)?u.x=i:c+=i;for(var h=0;h<c.length+64;h++)u.x^=c.charCodeAt(h)|0,h==c.length&&(u.d=u.x<<10^u.x>>>4),u.next()}function a(i,u){return u.x=i.x,u.y=i.y,u.z=i.z,u.w=i.w,u.v=i.v,u.d=i.d,u}function o(i,u){var c=new r(i),h=u&&u.state,p=function(){return(c.next()>>>0)/4294967296};return p.double=function(){do var f=c.next()>>>11,d=(c.next()>>>0)/4294967296,y=(f+d)/(1<<21);while(y===0);return y},p.int32=c.next,p.quick=p,h&&(typeof h=="object"&&a(h,c),p.state=function(){return a(c,{})}),p}n&&n.exports?n.exports=o:this.xorwow=o})(Se,e)})(Xs);var Ad=Xs.exports,Js={exports:{}};Js.exports;(function(e){(function(t,n,s){function r(i){var u=this;u.next=function(){var h=u.x,p=u.i,f,d;return f=h[p],f^=f>>>7,d=f^f<<24,f=h[p+1&7],d^=f^f>>>10,f=h[p+3&7],d^=f^f>>>3,f=h[p+4&7],d^=f^f<<7,f=h[p+7&7],f=f^f<<13,d^=f^f<<9,h[p]=d,u.i=p+1&7,d};function c(h,p){var f,d=[];if(p===(p|0))d[0]=p;else for(p=""+p,f=0;f<p.length;++f)d[f&7]=d[f&7]<<15^p.charCodeAt(f)+d[f+1&7]<<13;for(;d.length<8;)d.push(0);for(f=0;f<8&&d[f]===0;++f);for(f==8?d[7]=-1:d[f],h.x=d,h.i=0,f=256;f>0;--f)h.next()}c(u,i)}function a(i,u){return u.x=i.x.slice(),u.i=i.i,u}function o(i,u){i==null&&(i=+new Date);var c=new r(i),h=u&&u.state,p=function(){return(c.next()>>>0)/4294967296};return p.double=function(){do var f=c.next()>>>11,d=(c.next()>>>0)/4294967296,y=(f+d)/(1<<21);while(y===0);return y},p.int32=c.next,p.quick=p,h&&(h.x&&a(h,c),p.state=function(){return a(c,{})}),p}n&&n.exports?n.exports=o:this.xorshift7=o})(Se,e)})(Js);var Dd=Js.exports,Ys={exports:{}};Ys.exports;(function(e){(function(t,n,s){function r(i){var u=this;u.next=function(){var h=u.w,p=u.X,f=u.i,d,y;return u.w=h=h+1640531527|0,y=p[f+34&127],d=p[f=f+1&127],y^=y<<13,d^=d<<17,y^=y>>>15,d^=d>>>12,y=p[f]=y^d,u.i=f,y+(h^h>>>16)|0};function c(h,p){var f,d,y,T,N,S=[],I=128;for(p===(p|0)?(d=p,p=null):(p=p+"\0",d=0,I=Math.max(I,p.length)),y=0,T=-32;T<I;++T)p&&(d^=p.charCodeAt((T+32)%p.length)),T===0&&(N=d),d^=d<<10,d^=d>>>15,d^=d<<4,d^=d>>>13,T>=0&&(N=N+1640531527|0,f=S[T&127]^=d+N,y=f==0?y+1:0);for(y>=128&&(S[(p&&p.length||0)&127]=-1),y=127,T=4*128;T>0;--T)d=S[y+34&127],f=S[y=y+1&127],d^=d<<13,f^=f<<17,d^=d>>>15,f^=f>>>12,S[y]=d^f;h.w=N,h.X=S,h.i=y}c(u,i)}function a(i,u){return u.i=i.i,u.w=i.w,u.X=i.X.slice(),u}function o(i,u){i==null&&(i=+new Date);var c=new r(i),h=u&&u.state,p=function(){return(c.next()>>>0)/4294967296};return p.double=function(){do var f=c.next()>>>11,d=(c.next()>>>0)/4294967296,y=(f+d)/(1<<21);while(y===0);return y},p.int32=c.next,p.quick=p,h&&(h.X&&a(h,c),p.state=function(){return a(c,{})}),p}n&&n.exports?n.exports=o:this.xor4096=o})(Se,e)})(Ys);var Od=Ys.exports,Zs={exports:{}};Zs.exports;(function(e){(function(t,n,s){function r(i){var u=this,c="";u.next=function(){var p=u.b,f=u.c,d=u.d,y=u.a;return p=p<<25^p>>>7^f,f=f-d|0,d=d<<24^d>>>8^y,y=y-p|0,u.b=p=p<<20^p>>>12^f,u.c=f=f-d|0,u.d=d<<16^f>>>16^y,u.a=y-p|0},u.a=0,u.b=0,u.c=-1640531527,u.d=1367130551,i===Math.floor(i)?(u.a=i/4294967296|0,u.b=i|0):c+=i;for(var h=0;h<c.length+20;h++)u.b^=c.charCodeAt(h)|0,u.next()}function a(i,u){return u.a=i.a,u.b=i.b,u.c=i.c,u.d=i.d,u}function o(i,u){var c=new r(i),h=u&&u.state,p=function(){return(c.next()>>>0)/4294967296};return p.double=function(){do var f=c.next()>>>11,d=(c.next()>>>0)/4294967296,y=(f+d)/(1<<21);while(y===0);return y},p.int32=c.next,p.quick=p,h&&(typeof h=="object"&&a(h,c),p.state=function(){return a(c,{})}),p}n&&n.exports?n.exports=o:this.tychei=o})(Se,e)})(Zs);var Fd=Zs.exports,yo={exports:{}};const Cd={},Bd=Object.freeze(Object.defineProperty({__proto__:null,default:Cd},Symbol.toStringTag,{value:"Module"})),Rd=Mo(Bd);(function(e){(function(t,n,s){var r=256,a=6,o=52,i="random",u=s.pow(r,a),c=s.pow(2,o),h=c*2,p=r-1,f;function d(k,_,x){var D=[];_=_==!0?{entropy:!0}:_||{};var O=S(N(_.entropy?[k,A(n)]:k??I(),3),D),B=new y(D),L=function(){for(var R=B.g(a),G=u,j=0;R<c;)R=(R+j)*r,G*=r,j=B.g(1);for(;R>=h;)R/=2,G/=2,j>>>=1;return(R+j)/G};return L.int32=function(){return B.g(4)|0},L.quick=function(){return B.g(4)/4294967296},L.double=L,S(A(B.S),n),(_.pass||x||function(R,G,j,J){return J&&(J.S&&T(J,B),R.state=function(){return T(B,{})}),j?(s[i]=R,G):R})(L,O,"global"in _?_.global:this==s,_.state)}function y(k){var _,x=k.length,D=this,O=0,B=D.i=D.j=0,L=D.S=[];for(x||(k=[x++]);O<r;)L[O]=O++;for(O=0;O<r;O++)L[O]=L[B=p&B+k[O%x]+(_=L[O])],L[B]=_;(D.g=function(R){for(var G,j=0,J=D.i,st=D.j,$t=D.S;R--;)G=$t[J=p&J+1],j=j*r+$t[p&($t[J]=$t[st=p&st+G])+($t[st]=G)];return D.i=J,D.j=st,j})(r)}function T(k,_){return _.i=k.i,_.j=k.j,_.S=k.S.slice(),_}function N(k,_){var x=[],D=typeof k,O;if(_&&D=="object")for(O in k)try{x.push(N(k[O],_-1))}catch{}return x.length?x:D=="string"?k:k+"\0"}function S(k,_){for(var x=k+"",D,O=0;O<x.length;)_[p&O]=p&(D^=_[p&O]*19)+x.charCodeAt(O++);return A(_)}function I(){try{var k;return f&&(k=f.randomBytes)?k=k(r):(k=new Uint8Array(r),(t.crypto||t.msCrypto).getRandomValues(k)),A(k)}catch{var _=t.navigator,x=_&&_.plugins;return[+new Date,t,x,t.screen,A(n)]}}function A(k){return String.fromCharCode.apply(0,k)}if(S(s.random(),n),e.exports){e.exports=d;try{f=Rd}catch{}}else s["seed"+i]=d})(typeof self<"u"?self:Se,[],Math)})(yo);var Pd=yo.exports,Ld=xd,zd=Id,Vd=Ad,Ud=Dd,jd=Od,Wd=Fd,Ee=Pd;Ee.alea=Ld;Ee.xor128=zd;Ee.xorwow=Vd;Ee.xorshift7=Ud;Ee.xor4096=jd;Ee.tychei=Wd;var Qs=Ee;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class tr{constructor(t,n,s,r,a){this.mean=t,this.stdDev=n,this.dtype=s,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);const o=a||Math.random();this.random=Qs.alea(o.toString())}nextValue(){if(!isNaN(this.nextVal)){const r=this.nextVal;return this.nextVal=NaN,r}let t,n,s=!1;for(;!s;){let r,a,o;do r=2*this.random()-1,a=2*this.random()-1,o=r*r+a*a;while(o>=1||o===0);const i=Math.sqrt(-2*Math.log(o)/o);t=this.mean+this.stdDev*r*i,n=this.mean+this.stdDev*a*i,(!this.truncated||this.isValidTruncated(t))&&(s=!0)}return(!this.truncated||this.isValidTruncated(n))&&(this.nextVal=this.convertValue(n)),this.convertValue(t)}convertValue(t){return this.dtype==null||this.dtype==="float32"?t:Math.round(t)}isValidTruncated(t){return t<=this.upper&&t>=this.lower}}class qd{constructor(t,n,s,r){this.alpha=t,this.beta=1/n,this.dtype=s;const a=r||Math.random();this.randu=Qs.alea(a.toString()),this.randn=new tr(0,1,s,!1,this.randu()),t<1?this.d=t+2/3:this.d=t-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let t,n,s,r,a,o;for(;;){do r=this.randn.nextValue(),o=1+this.c*r;while(o<=0);if(o*=o*o,t=r*r,n=1-.331*t*t,s=.5*t+this.d*(1-o+Math.log(o)),a=this.randu(),a<n||Math.log(a)<s)break}return o=1/this.beta*this.d*o,this.alpha<1&&(o*=Math.pow(this.randu(),1/this.alpha)),this.convertValue(o)}convertValue(t){return this.dtype==="float32"?t:Math.round(t)}}class Gd{constructor(t=0,n=1,s,r){if(this.canReturnFloat=()=>this.dtype==null||this.dtype==="float32",this.min=t,this.range=n-t,this.dtype=s,r==null&&(r=Math.random()),typeof r=="number"&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${t} - ${n} <= 1 and dtype is not float`);this.random=Qs.alea(r)}convertValue(t){return this.canReturnFloat()?t:Math.round(t)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Md(e,t,n=1,s="float32",r){if(bt(e),n==null&&(n=1),s==null&&(s="float32"),s!=="float32"&&s!=="int32")throw new Error(`Unsupported data type ${s}`);const a=new qd(t,n,s,r),o=jt(e,s);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const Kd=b({randomGamma_:Md});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hd(e,t=0,n=1,s,r){if(bt(e),s!=null&&s==="bool")throw new Error(`Unsupported data type ${s}`);const a=new tr(t,n,s,!1,r),o=jt(e,s);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const bo=b({randomNormal_:Hd});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xd(e,t,n){if(t!=null&&t==="bool")throw new Error(`Unsupported data type ${t}`);return bo(e,0,1,t,n)}const Jd=b({randomStandardNormal_:Xd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yd(e,t=0,n=1,s="float32",r){bt(e);const a=jt(e,s),o=new Gd(t,n,null,r);for(let i=0;i<a.values.length;i++)a.values[i]=o.nextValue();return a.toTensor()}const er=b({randomUniform_:Yd});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zd(e,t,n,s){return er(e,t,n,"int32",s)}const Qd=b({randomUniformInt_:Zd});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function on(e,t,n=1,s="float32"){if(n===0)throw new Error("Cannot have a step of zero");const r={start:e,stop:t,step:n,dtype:s};return w.runKernel(dc,{},r)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tg(e){const n={input:m(e,"input","real")};return w.runKernel(gc,n)}const un=b({real_:tg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eg(e){const n={x:m(e,"x","reciprocal")};return w.runKernel(yc,n)}const ng=b({reciprocal_:eg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sg(e){const n={x:m(e,"x","relu")};return w.runKernel(bc,n)}const Xn=b({relu_:sg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rg(e){const n={x:m(e,"x","relu6")};return w.runKernel(Sc,n)}const wo=b({relu6_:rg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ag(e,t){const s={x:m(e,"x","reverse")},r={dims:t};return w.runKernel($c,s,r)}const we=b({reverse_:ag});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function og(e){const t=m(e,"x","reverse");return g(t.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${t.rank}.`),we(t,0)}const ig=b({reverse1d_:og});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ug(e,t){const n=m(e,"x","reverse");return g(n.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${n.rank}.`),we(n,t)}const cg=b({reverse2d_:ug});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lg(e,t){const n=m(e,"x","reverse");return g(n.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${n.rank}.`),we(n,t)}const hg=b({reverse3d_:lg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pg(e,t){const n=m(e,"x","reverse");return g(n.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${n.rank}.`),we(n,t)}const fg=b({reverse4d_:pg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mg(e){const n={x:m(e,"x","round")};return w.runKernel(kc,n)}const No=b({round_:mg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dg(e){const n={x:m(e,"x","rsqrt","float32")};return w.runKernel(Ec,n)}const gg=b({rsqrt_:dg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yg(e){const n={x:m(e,"x","selu")};return w.runKernel(Ac,n)}const bg=b({selu_:yg});function wg(e,t,n,s,r,a=[1,1],o="NHWC"){const i=m(e,"x","separableConv2d"),u=m(t,"depthwiseFilter","separableConv2d"),c=m(n,"pointwiseFilter","separableConv2d");let h=i,p=!1;if(i.rank===3&&(p=!0,h=$(i,[1,i.shape[0],i.shape[1],i.shape[2]])),o==="NCHW")throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");g(h.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${h.rank}.`),g(u.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${u.rank}.`),g(c.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${u.rank}.`),g(c.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${c.shape[0]}.`),g(c.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${c.shape[1]}.`);const f=u.shape[2],d=u.shape[3];g(c.shape[2]===f*d,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${f*d}, but got ${c.shape[2]}.`);const y=qs(h,u,s,r,o,a),N=Wn(y,c,1,"valid",o);return p?$(N,[N.shape[1],N.shape[2],N.shape[3]]):N}const Ng=b({separableConv2d_:wg});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Tg(e,t){const n=m(e,"x","setdiff1d"),s=m(t,"y","setdiff1d");g(n.dtype===s.dtype,()=>`x and y should have the same dtype, but got x (${n.dtype}) and y (${s.dtype}).`),g(n.rank===1,()=>`x should be 1D tensor, but got x (${n.shape}).`),g(s.rank===1,()=>`y should be 1D tensor, but got y (${s.shape}).`);const r=await n.data(),a=await s.data(),o=new Set(a);let i=0;for(let h=0;h<r.length;h++)o.has(r[h])||i++;const u=new is([i],n.dtype),c=new is([i],"int32");for(let h=0,p=0;h<r.length;h++)o.has(r[h])||(u.values[p]=r[h],c.values[p]=h,p++);return[u.toTensor(),c.toTensor()]}const Sg=Tg;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $g(e){const n={x:m(e,"x","sign")};return w.runKernel(Cc,n)}const kg=b({sign_:$g});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Eg(e){const n={x:m(e,"x","sin","float32")};return w.runKernel(Oc,n)}const vg=b({sin_:Eg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _g(e){const n={x:m(e,"x","sinh")};return w.runKernel(Fc,n)}const xg=b({sinh_:_g});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ig(e,t,n){const s=m(e,"x","slice1d");return g(s.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${s.rank} tensor`),q(s,[t],[n])}const Ag=b({slice1d_:Ig});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dg(e,t,n){const s=m(e,"x","slice2d");return g(s.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${s.rank} tensor`),q(s,t,n)}const Og=b({slice2d_:Dg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fg(e,t,n){const s=m(e,"x","slice3d");return g(s.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${s.rank} tensor`),q(s,t,n)}const Cg=b({slice3d_:Fg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bg(e,t,n){const s=m(e,"x","slice4d");return g(s.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${s.rank} tensor`),q(s,t,n)}const Rg=b({slice4d_:Bg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pg(e,t=-1){const n=m(e,"logits","softmax","float32");if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${t}`);const s={logits:n},r={dim:t};return w.runKernel(Uc,s,r)}const Lg=b({softmax_:Pg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zg(e){g(e.dtype==="complex64",()=>`The dtype for tf.spectral.fft() must be complex64 but got ${e.dtype}.`);const t={input:e};return w.runKernel(yu,t)}const nr=b({fft_:zg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vg(e){g(e.dtype==="complex64",()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${e.dtype}.`);const t={input:e};return w.runKernel(_u,t)}const Cn=b({ifft_:Vg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ug(e){const t=e.shape[e.shape.length-1],n=e.size/t;let s;if(t<=2){const r=$(e,[n,t]);s=Cn(r)}else{const r=[n,2*(t-1)],a=$(un(e),[n,t]),o=$(Kn(e),[n,t]),i=we(q(a,[0,1],[n,t-2]),1),u=v(we(q(o,[0,1],[n,t-2]),1),z(-1)),c=ct([a,i],1),h=ct([o,u],1),p=$(ee(c,h),[r[0],r[1]]);s=Cn(p)}if(s=un(s),e.rank===3&&e.shape[0]!==0){const r=s,a=e.shape[0];s=$(s,[a,s.shape[0]/a,s.shape[1]]),r.dispose()}return s}const To=b({irfft_:Ug});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jg(e,t,n=0){const r={x:m(e,"x","split")},a={numOrSizeSplits:t,axis:n};return w.runKernel(Vc,r,a)}const cn=b({split_:jg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wg(e,t){g(e.dtype==="float32",()=>`The dtype for rfft() must be real value but got ${e.dtype}`);let n=e.shape[e.shape.length-1];const s=e.size/n;let r;if(t!=null&&t<n){const y=e.shape.map(N=>0),T=e.shape.map(N=>N);T[e.shape.length-1]=t,r=q(e,y,T),n=t}else if(t!=null&&t>n){const y=e.shape.map(T=>T);y[e.shape.length-1]=t-n,r=ct([e,qe(y)],e.shape.length-1),n=t}else r=e;const a=Tt(r),o=$(ee(r,a),[s,n]),i=nr(o),u=Math.floor(n/2)+1,c=un(i),h=Kn(i),p=cn(c,[u,n-u],c.shape.length-1),f=cn(h,[u,n-u],h.shape.length-1),d=r.shape.slice();return d[r.shape.length-1]=u,$(ee(p[0],f[0]),d)}const sr=b({rfft_:Wg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qg(e,t){let n=m(e,"a","squaredDifference"),s=m(t,"b","squaredDifference");[n,s]=Q(n,s),nt(n.shape,s.shape);const r={a:n,b:s},a={};return w.runKernel(Kc,r,a)}const So=b({squaredDifference_:qg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gg(e,t){const n=m(e,"x","squeeze","string_or_numeric");return $(n,Yr(n.shape,t).newShape)}const rr=b({squeeze_:Gg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mg(e,t=0){const n=en(e,"tensors","stack","string_or_numeric");g(n.length>=1,()=>"Pass at least one tensor to tf.stack"),n.length>0&&g(t<=n[0].rank,()=>"Axis must be <= rank of the tensor");const s=n,r={axis:t};return w.runKernel(ic,s,r)}const Gt=b({stack_:Mg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kg(e,t=0){const s={x:m(e,"x","step")},r={alpha:t};return w.runKernel(ul,s,r)}const $o=b({step_:Kg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hg(e,t,n,s,r=0,a=0,o=0,i=0,u=0){const h={x:m(e,"x","stridedSlice","string_or_numeric")},p={begin:t,end:n,strides:s,beginMask:r,endMask:a,ellipsisMask:o,newAxisMask:i,shrinkAxisMask:u};return w.runKernel(Xc,h,p)}const Xg=b({stridedSlice_:Hg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jg(e){const n={x:m(e,"x","tan","float32")};return w.runKernel(tl,n)}const Yg=b({tan_:Jg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xt(e,t){$e(e);const n=se(e,t);if(n.length!==1)throw new Error("tensor1d() requires values to be a flat/TypedArray");return re(e,null,n,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ze(e,t,n){if($e(e),t!=null&&t.length!==2)throw new Error("tensor2d() requires shape to have two numbers");const s=se(e,n);if(s.length!==2&&s.length!==1)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return re(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ko(e,t,n){if($e(e),t!=null&&t.length!==3)throw new Error("tensor3d() requires shape to have three numbers");const s=se(e,n);if(s.length!==3&&s.length!==1)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return re(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zg(e,t,n){if($e(e),t!=null&&t.length!==4)throw new Error("tensor4d() requires shape to have four numbers");const s=se(e,n);if(s.length!==4&&s.length!==1)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return re(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qg(e,t,n){if($e(e),t!=null&&t.length!==5)throw new Error("tensor5d() requires shape to have five numbers");const s=se(e,n);if(s.length!==5&&s.length!==1)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return re(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ty(e,t,n){if($e(e),t!=null&&t.length!==6)throw new Error("tensor6d() requires shape to have six numbers");const s=se(e,n);if(s.length!==6&&s.length!==1)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return t=t||s,re(e,t,s,n)}function Eo(e,t,n){const s=t.rank>1?t.shape[t.rank-1]:1,r=t.rank>1?t.rank-1:1,a=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${s}, and batchDim: ${r}.`;if(n.rank<r)throw new Error(a+` update.rank < ${r}. `);if(e.length<s+(n.rank-r))throw new Error(a+` Output shape length < ${s+(n.rank-r)}`);if(n.rank!==r+e.length-s)throw new Error(a+` update.rank != ${r+e.length-s}`);for(let o=0;o<r;++o)if(n.shape[o]!==t.shape[o])throw new Error(a+` updates.shape[${o}] (${n.shape[o]}) != indices.shape[${o}] (${t.shape[o]}).`);for(let o=0;o<n.rank-r;++o)if(n.shape[o+r]!==e[o+s])throw new Error(a+` updates.shape[${o+r}] (${n.shape[o+r]}) != shape[${o+r}] (${e[o+r]})`)}function ar(e,t,n){if(t.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if(t.dtype!=="int32")throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(n.length===0){if(t.size===0)throw new Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(e.size===0)throw new Error(`Updates specified for empty output. updates shape: ${e.shape}`)}Eo(n,t,e)}function ey(e,t,n){const s=t.shape.length,r=s>1?t.shape[s-1]:1,a=n.length;let o=1;for(let p=r;p<a;++p)o*=n[p];const i=r<1?1:r,u=Z(t.shape)/i,c=[...pn(n.slice(0,r)),1],h=Z(n);return{sliceRank:r,numUpdates:u,sliceSize:o,strides:c,outputSize:h}}const GN=Object.freeze(Object.defineProperty({__proto__:null,calculateShapes:ey,validateInput:ar,validateUpdateShape:Eo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ny(e,t,n){const s=m(e,"tensor","tensorScatterupdate"),r=m(t,"indices","tensorScatterupdate","int32"),a=m(n,"updates","tensorScatterupdate");if(ar(a,r,s.shape),s.dtype!==a.dtype)throw new Error(`tensor and updates must have the same dtype, instead they are ${s.dtype} and ${a.dtype}.`);const o={tensor:s,indices:r,updates:a},i={};return w.runKernel(_c,o,i)}const sy=b({tensorScatterUpdate_:ny});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ry(e,t=1,n=!0){const s=m(e,"x","topk");if(s.rank===0)throw new Error("topk() expects the input to be of rank 1 or higher");const r=s.shape[s.shape.length-1];if(t<0)throw new Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>r)throw new Error(`'k' passed to topk() must be <= the last dimension (${r}) but got ${t}`);const a={x:s},o={k:t,sorted:n},[i,u]=w.runKernel(nl,a,o);return{values:i,indices:u}}const ay=b({topk_:ry});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oy(e,t=0,n=1,s,r){if(bt(e),s!=null&&s==="bool")throw new Error("Unsupported data type $ { dtype }");const a=new tr(t,n,s,!0,r),o=jt(e,s);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const iy=b({truncatedNormal_:oy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uy(e,t=0){const n=m(e,"x","unique","string_or_numeric");g(n.rank>0,()=>"The input tensor must be at least 1D");const s={x:n},r={axis:t},[a,o]=w.runKernel(rl,s,r);return{values:a,indices:o}}const cy=b({unique_:uy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ly(e,t,n){const s=m(e,"x","unsortedSegmentSum"),r=m(t,"segmentIds","unsortedSegmentSum","int32");g(Le(n),()=>"numSegments must be of dtype int");const a={x:s,segmentIds:r},o={numSegments:n};return w.runKernel(ol,a,o)}const hy=b({unsortedSegmentSum_:ly});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function py(e,t=0){const n=m(e,"x","unstack","string_or_numeric");g(t>=-n.shape.length&&t<n.shape.length,()=>`Axis = ${t} is not in [-${n.shape.length}, ${n.shape.length})`);const s={value:n},r={axis:t};return w.runKernel(al,s,r)}const ve=b({unstack_:py});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fy(e,t){return Ms(e,t,"right")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function my(e,t=!0,n,s){return w.makeVariable(e,t,n,s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dy(e,t){const n=[];for(let a=0;a<t.length;a++)t[a]&&n.push(a);const s=jt(e,"int32"),r=jt([n.length,e.length],"int32");for(let a=0;a<n.length;a++){const o=s.indexToLoc(n[a]),i=a*e.length;r.values.set(o,i)}return r.toTensor()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function gy(e){const t=m(e,"condition","whereAsync","bool"),n=await t.data(),s=dy(t.shape,n);return e!==t&&t.dispose(),s}const vo=gy;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function yy(e,t,n){const s=m(e,"tensor","boolMask"),r=m(t,"mask","boolMask","bool"),a=n??0,o=r.rank,i=s.shape;g(o>0,()=>"mask cannot be scalar"),lt(i.slice(a,a+o),r.shape,"mask's shape must match the first K dimensions of tensor's shape,");let u=1;for(let T=a;T<a+o;T++)u*=i[T];const c=i.slice(0,a).concat([u],i.slice(a+o)),h=$(s,c),p=$(r,[-1]),f=await vo(p),d=rr(f,[1]),y=so(h,d,a);return e!==s&&s.dispose(),t!==r&&r.dispose(),d.dispose(),h.dispose(),p.dispose(),f.dispose(),y}const by=yy;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wy(e,t,n){const s=m(e,"x","transpose");if(t==null&&(t=s.shape.map((o,i)=>i).reverse()),g(s.rank===t.length,()=>`Error in transpose: rank of input ${s.rank} must match length of perm ${t}.`),t.forEach(o=>{g(o>=0&&o<s.rank,()=>`All entries in 'perm' must be between 0 and ${s.rank-1} but got ${t}`)}),s.rank<=1)return s.clone();const r={x:s},a={perm:t};return s.dtype==="complex64"?V(()=>{let o=un(s),i=Kn(s);return o=w.runKernel(Yn,{x:o},a),i=w.runKernel(Yn,{x:i},a),n&&(i=Rt(i)),ee(o,i)}):w.runKernel(Yn,r,a)}const Ns=b({transpose_:wy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ny(e,t,n,s,r=!0){const a=m(e,"v","movingAverage"),o=m(t,"x","movingAverage"),i=m(n,"decay","movingAverage");va(a,o),g(Pt(a.shape,o.shape),()=>"Shape mismatch in v and x");const u=z(1),c=P(u,i);let h=v(P(o,a),c);if(r){g(s!=null,()=>"When using zeroDebias: true, step is required.");const p=m(s,"step","movingAverage");h=K(h,P(u,rn(i,p)))}return F(a,h)}const Ty=b({movingAverage_:Ny});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sy(e,t,n){bt(n);const s=m(e,"indices","scatterND","int32"),r=m(t,"updates","scatterND");ar(r,s,n);const a={indices:s,updates:r},o={shape:n};return w.runKernel(vc,a,o)}const $y=b({scatterND_:Sy});function ky(e,t,n,s){if(e.dtype!=="int32")throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${e.shape}.`);const r=e.rank>0?e.shape[0]:1,a=e.rank>1?e.shape[1]:1;if(n.length!==a)throw new Error(`outputShape has incorrect number of elements:, ${n.length}, should be: ${a}.`);const o=t.size;if(!(t.rank===0||t.rank===1&&o===r))throw new Error(`sparseValues has incorrect shape ${t.shape}, should be [] or [${r}]`);if(t.dtype!==s.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ey(e,t,n,s=0){bt(n);const r=m(e,"sparseIndices","sparseToDense","int32"),a=m(t,"sparseValues","sparseToDense","string_or_numeric"),o=m(s,"defaultValue","sparseToDense",a.dtype);ky(r,a,n,o);const i={sparseIndices:r,sparseValues:a,defaultValue:o},u={outputShape:n};return w.runKernel(Mc,i,u)}const vy=b({sparseToDense_:Ey});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _y(e,t){const n=m(t,"indices","gatherND","int32"),r={params:m(e,"x","gatherND","string_or_numeric"),indices:n};return w.runKernel(ku,r)}const xy=b({gatherND_:_y});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Iy(e,t){if(t==null)return e.shape.slice();if(Pt(e.shape,t))return t;if(e.shape.length===t.length){const n=[];for(let s=0;s<e.shape.length;s++)t[s]==null&&e.shape[s]!=null?n.push(e.shape[s]):n.push(t[s]);return n}return t}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ay(e,t,n,s){const r=m(e,"x","dropout");if(g(r.dtype==="float32",()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${r.dtype} tensor instead.`),g(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),t===0)return e instanceof et?r.clone():r;const a=Iy(r,n),o=1-t,i=K(no(F(er(a,0,1,"float32",s),o)),o);return v(r,i)}const Dy=b({dropout_:Ay});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _o(e){return Math.floor(Math.pow(2,Math.ceil(Math.log(e)/Math.log(2))))}function or(e,t,n){const s=1-e%2,r=new Float32Array(e);for(let a=0;a<e;++a){const o=2*Math.PI*a/(e+s-1);r[a]=t-n*Math.cos(o)}return xt(r,"float32")}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Oy(e,t,n=1){const s=m(e,"predictions","inTopK"),r=m(t,"targets","inTopK");g(s.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${s.rank}`),g(s.rank-1===r.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${s.rank} and targets rank ${r.rank}`),lt(s.shape.slice(0,s.shape.length-1),r.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const a=s.shape[s.shape.length-1];g(n>0&&n<=a,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${a}), but got ${n}`);const o=await s.data(),i=await r.data(),[u,c]=[o.length/a,a],h=Zr("bool",u);for(let p=0;p<u;p++){const f=p*c,d=o.subarray(f,f+c),y=[];for(let T=0;T<d.length;T++)y.push({value:d[T],index:T});y.sort((T,N)=>N.value-T.value),h[p]=0;for(let T=0;T<n;T++)if(y[T].index===i[p]){h[p]=1;break}}return e!==s&&s.dispose(),t!==r&&r.dispose(),Bt(h,r.shape,"bool")}const Fy=Oy;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cy(e,t,n,s,r,a="NHWC",o){let i=e;e.rank===3&&(i=$(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let u=t;u.rank===3&&(u=$(t,[1,t.shape[0],t.shape[1],t.shape[2]])),g(i.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${i.shape}.`),g(u.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${u.shape}.`),g(n.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);const c=a==="NHWC"?i.shape[3]:i.shape[1],h=a==="NHWC"?u.shape[3]:u.shape[1];g(c===n[2],()=>`Error in conv2dDerFilter: depth of input ${c}) must match input depth in filter (${n[2]}.`),g(h===n[3],()=>`Error in conv2dDerFilter: depth of dy (${h}) must match output depth for filter (${n[3]}).`),Ft("conv2dDerFilter",r,o);const p={x:i,dy:u},f={strides:s,pad:r,dataFormat:a,dimRoundingMode:o,filterShape:n};return w.runKernel(Ki,p,f)}const By=b({conv2DBackpropFilter_:Cy});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ir(e,t,n){if(n==null||n==="linear")return e;if(n==="relu")return v(e,$o(t));throw new Error(`Cannot compute gradient for fused activation ${n}.`)}function ur(e,t){let n=t;const s=Ja(e.shape,t.shape);return s.length>0&&(n=M(n,s)),$(n,e.shape)}function cr(e,t,n,s){if(t==="linear")return e;if(t==="relu")return Xn(e);if(t==="elu")return Za(e);if(t==="relu6")return wo(e);if(t==="prelu")return go(e,n);if(t==="leakyrelu")return ao(e,s);if(t==="sigmoid")return Re(e);throw new Error(`Unknown fused activation ${t}.`)}const lr=(e,t)=>!(e>0)||t==="linear";/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ry({x:e,filter:t,strides:n,pad:s,dataFormat:r="NHWC",dilations:a=[1,1],dimRoundingMode:o,bias:i,activation:u="linear",preluActivationWeights:c,leakyreluAlpha:h}){if(u=u||"linear",lr(w.state.gradientDepth,u)===!1){g(r==="NHWC",()=>`Error in fused conv2d: got dataFormat of ${r} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let x=Wn(e,t,n,s,r,a,o);return i!=null&&(x=F(x,i)),cr(x,u,c,h)}const p=m(e,"x","conv2d","float32"),f=m(t,"filter","conv2d","float32");let d=p,y=!1;p.rank===3&&(y=!0,d=$(p,[1,p.shape[0],p.shape[1],p.shape[2]])),g(d.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${d.rank}.`),g(f.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${f.rank}.`),Ft("fused conv2d",s,o);const T=r==="NHWC"?d.shape[3]:d.shape[1];g(f.shape[2]===T,()=>`Error in conv2d: depth of input (${T}) must match input depth for filter ${f.shape[2]}.`),g(ae(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`);const N=Un(d.shape,f.shape,n,a,s,o);let S;i!=null&&(S=m(i,"bias","fused conv2d"),[S]=Q(S,p),r==="NHWC"?nt(N.outShape,S.shape):(g(S.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${S.shape.length}.`),g(S.shape.length===0||S.shape[0]===N.outChannels||S.shape[0]===1,()=>`Error in fused conv2d: bias shape (${S.shape}) is not compatible with the number of output channels (${N.outChannels})`)));let I;if(c!=null){const x=c.shape;if(g(x.length<=1||x.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${x.length}.`),x.length===1)g(x[0]===1||x[0]===N.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${x}) is not compatible with the number of output channels (${N.outChannels}).`);else if(x.length===3)try{nt(x,N.outShape)}catch{const O=`Error in fused conv2d: PReLU activation weights (${x}) is not compatible with the output shape of the conv2d (${N.outShape}).`;throw Error(O)}I=m(c,"prelu weights","fused conv2d")}const A=(x,D)=>{g(r==="NHWC",()=>`Error in gradient of fused conv2D: got dataFormat of ${r} but only NHWC is currently supported.`);const[O,B,L,R]=D,G=ir(x,L,u);g(An(a),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${a}'`);const j=Xa(B.shape,G,O,n,s),J=By(B,G,O.shape,n,s),st=[j,J];if(R!=null){const $t=ur(R,G);st.push($t)}return st},k={x:d,filter:f,bias:S,preluActivationWeights:I},_={strides:n,pad:s,dataFormat:r,dilations:a,dimRoundingMode:o,activation:u,leakyreluAlpha:h};return i==null?qt((D,O,B)=>{let L=w.runKernel(wr,k,_);return B([O,D,L]),y&&(L=$(L,[L.shape[1],L.shape[2],L.shape[3]])),{value:L,gradFunc:A}})(d,f):qt((D,O,B,L)=>{let R=w.runKernel(wr,k,_);return L([O,D,R,B]),y&&(R=$(R,[R.shape[1],R.shape[2],R.shape[3]])),{value:R,gradFunc:A}})(d,f,S)}const Py=b({fusedConv2d_:Ry});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ly(e,t,n,s,r,a=[1,1],o){let i=e;e.rank===3&&(i=$(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let u=t;u.rank===3&&(u=$(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const c={x:i,dy:u},h={strides:s,pad:r,dimRoundingMode:o,dilations:a,filterShape:n};return w.runKernel(au,c,h)}const zy=b({depthwiseConv2dNativeBackpropFilter_:Ly});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vy(e,t,n,s,r,a=[1,1],o){let i=t,u=!1;t.rank===3&&(u=!0,i=$(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const c={dy:i,filter:n},h={strides:s,pad:r,dimRoundingMode:o,dilations:a,inputShape:e},p=w.runKernel(ou,c,h);return u?$(p,[p.shape[1],p.shape[2],p.shape[3]]):p}const Uy=b({depthwiseConv2dNativeBackpropInput_:Vy});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jy({x:e,filter:t,strides:n,pad:s,dataFormat:r="NHWC",dilations:a=[1,1],dimRoundingMode:o,bias:i,activation:u="linear",preluActivationWeights:c,leakyreluAlpha:h}){if(lr(w.state.gradientDepth,u)===!1){let _=qs(e,t,n,s,r,a,o);return i!=null&&(_=F(_,i)),cr(_,u,c,h)}const p=m(e,"x","depthwiseConv2d","float32"),f=m(t,"filter","depthwiseConv2d","float32");let d=p,y=!1;p.rank===3&&(y=!0,d=$(p,[1,p.shape[0],p.shape[1],p.shape[2]])),g(d.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${d.rank}.`),g(f.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${f.rank}.`),g(d.shape[3]===f.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${d.shape[3]}) must match the inChannels dimension in filter ${f.shape[2]}.`),a==null&&(a=[1,1]),g(ae(n,a),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),Ft("fused depthwiseConv2d",s,o);const T=Un(d.shape,f.shape,n,a,s,o,!0);let N;i!=null&&(N=m(i,"bias","fused conv2d"),[N]=Q(N,p),nt(T.outShape,N.shape));let S;c!=null&&(S=m(c,"prelu weights","fused depthwiseConv2d"));const I=(_,x)=>{g(An(a),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${a}'`);const[D,O,B,L]=x,R=ir(_,B,u),G=Uy(O.shape,R,D,n,s,a,o),j=zy(O,R,D.shape,n,s,a,o);if(L!=null){const J=ur(N,R);return[G,j,J]}return[G,j]},A={x:d,filter:f,bias:N,preluActivationWeights:S},k={strides:n,pad:s,dataFormat:r,dilations:a,dimRoundingMode:o,activation:u,leakyreluAlpha:h};return i==null?qt((x,D,O)=>{let B=w.runKernel(Nr,A,k);return O([D,x,B]),y&&(B=$(B,[B.shape[1],B.shape[2],B.shape[3]])),{value:B,gradFunc:I}})(d,f):qt((x,D,O,B)=>{let L=w.runKernel(Nr,A,k);return B([D,x,L,O]),y&&(L=$(L,[L.shape[1],L.shape[2],L.shape[3]])),{value:L,gradFunc:I}})(d,f,N)}const Wy=b({fusedDepthwiseConv2d_:jy});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qy({a:e,b:t,transposeA:n=!1,transposeB:s=!1,bias:r,activation:a="linear",preluActivationWeights:o,leakyreluAlpha:i=.2}){if(lr(w.state.gradientDepth,a)===!1){let R=U(e,t,n,s);return r!=null&&(R=F(R,r)),cr(R,a,o,i)}let u=m(e,"a","fused matMul"),c=m(t,"b","fused matMul");[u,c]=Q(u,c);const h=n?u.shape[u.rank-2]:u.shape[u.rank-1],p=s?c.shape[c.rank-1]:c.shape[c.rank-2],f=n?u.shape[u.rank-1]:u.shape[u.rank-2],d=s?c.shape[c.rank-2]:c.shape[c.rank-1],y=u.shape.slice(0,-2),T=c.shape.slice(0,-2),N=Z(y),S=Z(T);g(h===p,()=>`Error in fused matMul: inner shapes (${h}) and (${p}) of Tensors with shapes ${u.shape} and ${c.shape} and transposeA=${n} and transposeB=${s} must match.`);const A=nt(u.shape.slice(0,-2),c.shape.slice(0,-2)).concat([f,d]),k=n?$(u,[N,h,f]):$(u,[N,f,h]),_=s?$(c,[S,d,p]):$(c,[S,p,d]);let x;r!=null&&(x=m(r,"bias","fused matMul"),[x]=Q(x,u),nt(A,x.shape));let D;o!=null&&(D=m(o,"prelu weights","fused matMul"));const O=(R,G)=>{const[j,J,st,$t]=G,Lt=ir($(R,st.shape),st,a);let xe,Ie;if(!n&&!s?(xe=U(Lt,J,!1,!0),Ie=U(j,Lt,!0,!1)):!n&&s?(xe=U(Lt,J,!1,!1),Ie=U(Lt,j,!0,!1)):n&&!s?(xe=U(J,Lt,!1,!0),Ie=U(j,Lt,!1,!1)):(xe=U(J,Lt,!0,!0),Ie=U(Lt,j,!0,!0)),r!=null){const qo=ur($t,Lt);return[xe,Ie,qo]}else return[xe,Ie]},B={a:k,b:_,bias:x,preluActivationWeights:D},L={transposeA:n,transposeB:s,activation:a,leakyreluAlpha:i};return r==null?qt((G,j,J)=>{const st=w.runKernel(br,B,L);return J([G,j,st]),{value:$(st,A),gradFunc:O}})(k,_):qt((G,j,J,st)=>{const $t=w.runKernel(br,B,L);return st([G,j,$t,J]),{value:$($t,A),gradFunc:O}})(k,_,x)}const Gy=b({fusedMatMul_:qy});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const My=Object.freeze(Object.defineProperty({__proto__:null,conv2d:Py,depthwiseConv2d:Wy,matMul:Gy},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ky(e){return or(e,.54,.46)}const Hy=b({hammingWindow_:Ky});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xy(e){return or(e,.5,.5)}const xo=b({hannWindow_:Xy});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jy(e,t,n,s=!1,r=0){let a=0;const o=[];for(;a+t<=e.size;)o.push(q(e,a,t)),a+=n;if(s)for(;a<e.size;){const i=a+t-e.size,u=ct([q(e,a,t-i),dn([i],r)]);o.push(u),a+=n}return o.length===0?Ze([],[0,t]):$(ct(o),[o.length,t])}const Io=b({frame_:Jy});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yy(e,t,n,s,r=xo){s==null&&(s=_o(t));const a=Io(e,t,n),o=v(a,r(t));return sr(o,s)}const Zy=b({stft_:Yy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qy(e,t,n,s,r="bilinear",a=0){const o=m(e,"image","cropAndResize"),i=m(t,"boxes","cropAndResize","float32"),u=m(n,"boxInd","cropAndResize","int32"),c=i.shape[0];g(o.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${o.rank}.`),g(i.rank===2&&i.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${c},4] but had shape ${i.shape}.`),g(u.rank===1&&u.shape[0]===c,()=>`Error in cropAndResize: boxInd must be have size [${c}] but had shape ${i.shape}.`),g(s.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${s.length}.`),g(s[0]>=1&&s[1]>=1,()=>`cropSize must be atleast [1,1], but was ${s}`),g(r==="bilinear"||r==="nearest",()=>`method must be bilinear or nearest, but was ${r}`);const h={image:o,boxes:i,boxInd:u},p={method:r,extrapolationValue:a,cropSize:s};return w.runKernel(eu,h,p)}const tb=b({cropAndResize_:Qy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eb(e){const t=m(e,"image","flipLeftRight","float32");g(t.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);const n={image:t};return w.runKernel(wu,n,{})}const nb=b({flipLeftRight_:eb});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sb(e){const t=m(e,"image","grayscaleToRGB"),n=t.rank-1,s=t.shape[n];g(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),g(s===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${s}.`);const r=new Array(t.rank);return r.fill(1,0,n),r[n]=3,Ye(t,r)}const rb=b({grayscaleToRGB_:sb});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ab(e){const t=m(e,"image","RGBToGrayscale"),n=t.rank-1,s=t.shape[n];g(t.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${t.rank}.`),g(s===3,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${s}.`);const r=t.dtype,a=tt(t,"float32"),o=xt([.2989,.587,.114]);let i;switch(t.rank){case 2:i=Oe("ij,j->i",a,o);break;case 3:i=Oe("ijk,k->ij",a,o);break;case 4:i=Oe("ijkl,l->ijk",a,o);break;case 5:i=Oe("ijklm,m->ijkl",a,o);break;case 6:i=Oe("ijklmn,n->ijklm",a,o);break;default:throw new Error("Not a valid tensor rank.")}return i=Kt(i,-1),tt(i,r)}const ob=b({rgbToGrayscale_:ab});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ib(e,t,n=0,s=.5){const r=m(e,"image","rotateWithOffset","float32");g(r.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${r.rank}.`);const a={image:r},o={radians:t,fillValue:n,center:s};return w.runKernel(cl,a,o)}const ub=b({rotateWithOffset_:ib});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ge(e,t,n,s,r,a){s==null&&(s=.5),r==null&&(r=Number.NEGATIVE_INFINITY),a==null&&(a=0);const o=e.shape[0];return n=Math.min(n,o),g(0<=s&&s<=1,()=>`iouThreshold must be in [0, 1], but was '${s}'`),g(e.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${e.rank}'`),g(e.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${e.shape[1]}`),g(t.rank===1,()=>"scores must be a 1D tensor"),g(t.shape[0]===o,()=>`scores has incompatible shape with boxes. Expected ${o}, but was ${t.shape[0]}`),g(0<=a&&a<=1,()=>`softNmsSigma must be in [0, 1], but was '${a}'`),{maxOutputSize:n,iouThreshold:s,scoreThreshold:r,softNmsSigma:a}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cb(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY){const a=m(e,"boxes","nonMaxSuppression","float32"),o=m(t,"scores","nonMaxSuppression","float32"),i=Ge(a,o,n,s,r);n=i.maxOutputSize,s=i.iouThreshold,r=i.scoreThreshold;const u={maxOutputSize:n,iouThreshold:s,scoreThreshold:r};return w.runKernel(nc,{boxes:a,scores:o},u)}const lb=b({nonMaxSuppression_:cb});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hb(e,t,n){const s=pb(e,t,n),r=s<0?-(s+1):s;e.splice(r,0,t)}function pb(e,t,n){return mb(e,t,n||fb)}function fb(e,t){return e>t?1:e<t?-1:0}function mb(e,t,n){let s=0,r=e.length,a=0,o=!1;for(;s<r;){a=s+(r-s>>>1);const i=n(t,e[a]);i>0?s=a+1:(r=a,o=!i)}return o?s:-s-1}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function db(e,t,n,s,r){return hr(e,t,n,s,r,0)}function gb(e,t,n,s,r,a){return hr(e,t,n,s,r,0,!1,a,!0)}function yb(e,t,n,s,r,a){return hr(e,t,n,s,r,a,!0)}function hr(e,t,n,s,r,a,o=!1,i=!1,u=!1){const c=[];for(let N=0;N<t.length;N++)t[N]>r&&c.push({score:t[N],boxIndex:N,suppressBeginIndex:0});c.sort(Br);const h=a>0?-.5/a:0,p=[],f=[];for(;p.length<n&&c.length>0;){const N=c.pop(),{score:S,boxIndex:I,suppressBeginIndex:A}=N;if(S<r)break;let k=!1;for(let _=p.length-1;_>=A;--_){const x=bb(e,I,p[_]);if(x>=s){k=!0;break}if(N.score=N.score*wb(s,h,x),N.score<=r)break}N.suppressBeginIndex=p.length,k||(N.score===S?(p.push(I),f.push(N.score)):N.score>r&&hb(c,N,Br))}const d=p.length,y=n-d;i&&y>0&&(p.push(...new Array(y).fill(0)),f.push(...new Array(y).fill(0)));const T={selectedIndices:p};return o&&(T.selectedScores=f),u&&(T.validOutputs=d),T}function bb(e,t,n){const s=e.subarray(t*4,t*4+4),r=e.subarray(n*4,n*4+4),a=Math.min(s[0],s[2]),o=Math.min(s[1],s[3]),i=Math.max(s[0],s[2]),u=Math.max(s[1],s[3]),c=Math.min(r[0],r[2]),h=Math.min(r[1],r[3]),p=Math.max(r[0],r[2]),f=Math.max(r[1],r[3]),d=(i-a)*(u-o),y=(p-c)*(f-h);if(d<=0||y<=0)return 0;const T=Math.max(a,c),N=Math.max(o,h),S=Math.min(i,p),I=Math.min(u,f),A=Math.max(S-T,0)*Math.max(I-N,0);return A/(d+y-A)}function wb(e,t,n){const s=Math.exp(t*n*n);return n<=e?s:0}function Br(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Nb(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY){const a=m(e,"boxes","nonMaxSuppressionAsync"),o=m(t,"scores","nonMaxSuppressionAsync"),i=Ge(a,o,n,s,r);n=i.maxOutputSize,s=i.iouThreshold,r=i.scoreThreshold;const u=await Promise.all([a.data(),o.data()]),c=u[0],h=u[1],{selectedIndices:p}=db(c,h,n,s,r);return a!==e&&a.dispose(),o!==t&&o.dispose(),xt(p,"int32")}const Tb=Nb;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sb(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,a=0){const o=m(e,"boxes","nonMaxSuppression"),i=m(t,"scores","nonMaxSuppression"),u=Ge(o,i,n,s,r,a);n=u.maxOutputSize,s=u.iouThreshold,r=u.scoreThreshold,a=u.softNmsSigma;const c={boxes:o,scores:i},h={maxOutputSize:n,iouThreshold:s,scoreThreshold:r,softNmsSigma:a},p=w.runKernel(rc,c,h);return{selectedIndices:p[0],selectedScores:p[1]}}const $b=b({nonMaxSuppressionWithScore_:Sb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function kb(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,a=0){const o=m(e,"boxes","nonMaxSuppressionAsync"),i=m(t,"scores","nonMaxSuppressionAsync"),u=Ge(o,i,n,s,r,a);n=u.maxOutputSize,s=u.iouThreshold,r=u.scoreThreshold,a=u.softNmsSigma;const c=await Promise.all([o.data(),i.data()]),h=c[0],p=c[1],{selectedIndices:f,selectedScores:d}=yb(h,p,n,s,r,a);return o!==e&&o.dispose(),i!==t&&i.dispose(),{selectedIndices:xt(f,"int32"),selectedScores:xt(d)}}const Eb=kb;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vb(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,a=!1){const o=m(e,"boxes","nonMaxSuppression"),i=m(t,"scores","nonMaxSuppression"),u=Ge(o,i,n,s,r,null),c=u.maxOutputSize,h=u.iouThreshold,p=u.scoreThreshold,f={boxes:o,scores:i},d={maxOutputSize:c,iouThreshold:h,scoreThreshold:p,padToMaxOutputSize:a},y=w.runKernel(sc,f,d);return{selectedIndices:y[0],validOutputs:y[1]}}const _b=b({nonMaxSuppressionPadded_:vb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function xb(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,a=!1){const o=m(e,"boxes","nonMaxSuppressionAsync"),i=m(t,"scores","nonMaxSuppressionAsync"),u=Ge(o,i,n,s,r,null),c=u.maxOutputSize,h=u.iouThreshold,p=u.scoreThreshold,[f,d]=await Promise.all([o.data(),i.data()]),{selectedIndices:y,validOutputs:T}=gb(f,d,c,h,p,a);return o!==e&&o.dispose(),i!==t&&i.dispose(),{selectedIndices:xt(y,"int32"),validOutputs:z(T,"int32")}}const Ib=xb;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ab(e,t,n=!1,s=!1){const r=m(e,"images","resizeBilinear");g(r.rank===3||r.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${r.rank}.`),g(t.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),g(s===!1||n===!1,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let a=r,o=!1;r.rank===3&&(o=!0,a=$(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const i={images:a},u={alignCorners:n,halfPixelCenters:s,size:t},c=w.runKernel(Tc,i,u);return o?$(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const Db=b({resizeBilinear_:Ab});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ob(e,t,n=!1,s=!1){const r=m(e,"images","resizeNearestNeighbor");g(r.rank===3||r.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${r.rank}.`),g(t.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),g(r.dtype==="float32"||r.dtype==="int32",()=>"`images` must have `int32` or `float32` as dtype"),g(s===!1||n===!1,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let a=r,o=!1;r.rank===3&&(o=!0,a=$(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const i={images:a},u={alignCorners:n,halfPixelCenters:s,size:t},c=w.runKernel(Nc,i,u);return o?$(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const Fb=b({resizeNearestNeighbor_:Ob});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cb(e,t="binary",n=!1,s=.5){const r=m(e,"image","threshold"),a=.2989,o=.587,i=.114,u=r.shape[0]*r.shape[1];let c=v(xt([s]),255),h,p,f,d;if(g(r.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${r.rank}.`),g(r.shape[2]===3||r.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${r.shape[2]}.`),g(r.dtype==="int32"||r.dtype==="float32",()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${r.dtype}.`),g(t==="otsu"||t==="binary",()=>`Method must be binary or otsu, but was ${t}`),r.shape[2]===3){[h,p,f]=cn(r,[1,1,1],-1);const N=v(h,a),S=v(p,o),I=v(f,i);d=F(F(N,S),I)}else d=e;if(t==="otsu"){const N=Ha(tt(No(d),"int32"),Bt([]),256);c=Bb(N,u)}const y=n?Gs(d,c):Mn(d,c);return tt(v(y,255),"int32")}function Bb(e,t){let n=xt([-1]),s=xt([0]),r=xt([0]),a,o,i,u,c,h;for(let p=0;p<e.size-1;p++){a=q(e,0,p+1),o=q(e,p+1),c=K(M(a),t),h=K(M(o),t);const f=M(v(a,on(0,a.size)));i=K(f,M(a));const d=dn(o.shape,a.size),y=F(on(0,o.size),d),T=v(o,y);u=K(M(T),M(o));const N=P(i,u),S=P(i,u),I=v(c,h);r=v(v(I,N),S);const A=Mn(r,s);s=Zt(A,r,s),n=Zt(A,xt([p]),n)}return n}const Rb=b({threshold_:Cb});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pb(e,t,n="nearest",s="constant",r=0,a){const o=m(e,"image","transform","float32"),i=m(t,"transforms","transform","float32");g(o.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${o.rank}.`),g(i.rank===2&&(i.shape[0]===o.shape[0]||i.shape[0]===1)&&i.shape[1]===8,()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),g(a==null||a.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${a}.`);const u={image:o,transforms:i},c={interpolation:n,fillMode:s,fillValue:r,outputShape:a};return w.runKernel(sl,u,c)}const Lb=b({transform_:Pb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zb(e,t,n){const s=m(e,"a","bandPart");g(s.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${s.rank}.`);const r=s.shape,[a,o]=s.shape.slice(-2);let i,u;typeof t=="number"?(g(t%1===0,()=>`bandPart(): numLower must be an integer, got ${t}.`),g(t<=a,()=>`bandPart(): numLower (${t}) must not be greater than the number of rows (${a}).`),i=m(t<0?a:t,"numLower","bandPart")):(g(t.dtype==="int32",()=>"bandPart(): numLower's dtype must be an int32."),i=Zt(ws(t,0),a,Fn(t,a))),typeof n=="number"?(g(n%1===0,()=>`bandPart(): numUpper must be an integer, got ${n}.`),g(n<=o,()=>`bandPart(): numUpper (${n}) must not be greater than the number of columns (${o}).`),u=m(n<0?o:n,"numUpper","bandPart")):(g(n.dtype==="int32",()=>"bandPart(): numUpper's dtype must be an int32."),u=Zt(ws(n,0),o,Fn(n,o)));const c=$(on(0,a,1,"int32"),[-1,1]),h=on(0,o,1,"int32"),p=P(c,h),f=Dn(Gs(p,i),ro(p,Rt(u))),d=qe([a,o],s.dtype);return $(Gt(ve($(s,[-1,a,o])).map(y=>Zt(f,y,d))),r)}const Vb=b({bandPart_:zb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ub(e){let t;if(Array.isArray(e)){t=!1,g(e!=null&&e.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const r=e[0].shape[0];for(let a=1;a<e.length;++a)g(e[a].shape[0]===r,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${e[a].shape[0]} vs. ${r})`)}else t=!0,e=cn(e,e.shape[0],0).map(r=>rr(r,[0]));g(e.length<=e[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${e.length}) exceeds number of dimensions (${e[0].shape[0]}).`);const n=[],s=e;for(let r=0;r<e.length;++r)n.push(w.tidy(()=>{let a=s[r];if(r>0)for(let o=0;o<r;++o){const i=v(M(v(n[o],a)),n[o]);a=P(a,i)}return K(a,Gn(a,"euclidean"))}));return t?Gt(n,0):n}const jb=b({gramSchmidt_:Ub});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wb(e,t=!1){if(g(e.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${e.rank}`),e.rank===2)return Rr(e,t);{const n=e.shape.slice(0,e.shape.length-2).reduce((u,c)=>u*c),s=ve($(e,[n,e.shape[e.shape.length-2],e.shape[e.shape.length-1]]),0),r=[],a=[];s.forEach(u=>{const[c,h]=Rr(u,t);r.push(c),a.push(h)});const o=$(Gt(r,0),e.shape),i=$(Gt(a,0),e.shape);return[o,i]}}function Rr(e,t=!1){return w.tidy(()=>{g(e.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${e.shape.length}D Tensor.`);const n=e.shape[0],s=e.shape[1];let r=eo(n),a=Yt(e);const o=Ze([[1]],[1,1]);let i=Yt(o);const u=n>=s?s:n;for(let c=0;c<u;++c){const h=a,p=i,f=r;[i,a,r]=w.tidy(()=>{const d=q(a,[c,c],[n-c,1]),y=Gn(d),T=q(a,[c,c],[1,1]),N=Zt(Mn(T,0),Ze([[-1]]),Ze([[1]])),S=P(T,v(N,y)),I=K(d,S);I.shape[0]===1?i=Yt(o):i=ct([o,q(I,[1,0],[I.shape[0]-1,I.shape[1]])],0);const A=Rt(K(U(N,S),y)),k=q(a,[c,0],[n-c,s]),_=v(A,i),x=Ns(i);if(c===0)a=P(k,U(_,U(x,k)));else{const B=P(k,U(_,U(x,k)));a=ct([q(a,[0,0],[c,s]),B],0)}const D=Ns(_),O=q(r,[0,c],[n,r.shape[1]-c]);if(c===0)r=P(O,U(U(O,i),D));else{const B=P(O,U(U(O,i),D));r=ct([q(r,[0,0],[n,c]),B],1)}return[i,a,r]}),pt([h,p,f])}return!t&&n>s&&(r=q(r,[0,0],[n,s]),a=q(a,[0,0],[s,s])),[r,a]})}const qb=b({qr_:Wb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var ht;(function(e){e[e.NONE=0]="NONE",e[e.MEAN=1]="MEAN",e[e.SUM=2]="SUM",e[e.SUM_BY_NONZERO_WEIGHTS=3]="SUM_BY_NONZERO_WEIGHTS"})(ht||(ht={}));function Gb(e,t,n=ht.SUM_BY_NONZERO_WEIGHTS){const s=m(e,"losses","computeWeightedLoss");let r=null;t!=null&&(r=m(t,"weights","computeWeightedLoss"));const a=r==null?s:v(s,r);if(n===ht.NONE)return a;if(n===ht.SUM)return M(a);if(n===ht.MEAN){if(r==null)return On(a);{const o=s.size/r.size,i=K(M(a),M(r));return o>1?K(i,z(o)):i}}if(n===ht.SUM_BY_NONZERO_WEIGHTS){if(r==null)return K(M(a),z(s.size));{const o=v(r,fe(s.shape)),i=tt(M(fo(o,z(0))),"float32");return K(M(a),i)}}throw Error(`Unknown reduction: ${n}`)}const Mt=b({computeWeightedLoss_:Gb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mb(e,t,n,s=ht.SUM_BY_NONZERO_WEIGHTS){const r=m(e,"labels","absoluteDifference"),a=m(t,"predictions","absoluteDifference");let o=null;n!=null&&(o=m(n,"weights","absoluteDifference")),lt(r.shape,a.shape,"Error in absoluteDifference: ");const i=Nt(P(r,a));return Mt(i,o,s)}const Kb=b({absoluteDifference_:Mb});function Hb(e,t,n,s,r=ht.SUM_BY_NONZERO_WEIGHTS){const a=m(e,"labels","cosineDistance"),o=m(t,"predictions","cosineDistance");let i=null;s!=null&&(i=m(s,"weights","cosineDistance")),lt(a.shape,o.shape,"Error in cosineDistance: ");const u=z(1),c=P(u,M(v(a,o),n,!0));return Mt(c,i,r)}const Xb=b({cosineDistance_:Hb});function Jb(e,t,n,s=ht.SUM_BY_NONZERO_WEIGHTS){let r=m(e,"labels","hingeLoss");const a=m(t,"predictions","hingeLoss");let o=null;n!=null&&(o=m(n,"weights","hingeLoss")),lt(r.shape,a.shape,"Error in hingeLoss: ");const i=z(1);r=P(v(z(2),r),i);const u=Xn(P(i,v(r,a)));return Mt(u,o,s)}const Yb=b({hingeLoss_:Jb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zb(e,t,n,s=1,r=ht.SUM_BY_NONZERO_WEIGHTS){const a=m(e,"labels","huberLoss"),o=m(t,"predictions","huberLoss");let i=null;n!=null&&(i=m(n,"weights","huberLoss")),lt(a.shape,o.shape,"Error in huberLoss: ");const u=z(s),c=Nt(P(o,a)),h=Fn(c,u),p=P(c,h),f=F(v(z(.5),At(h)),v(u,p));return Mt(f,i,r)}const Qb=b({huberLoss_:Zb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tw(e,t,n,s=1e-7,r=ht.SUM_BY_NONZERO_WEIGHTS){const a=m(e,"labels","logLoss"),o=m(t,"predictions","logLoss");let i=null;n!=null&&(i=m(n,"weights","logLoss")),lt(a.shape,o.shape,"Error in logLoss: ");const u=z(1),c=z(s),h=Rt(v(a,an(F(o,c)))),p=v(P(u,a),an(F(P(u,o),c))),f=P(h,p);return Mt(f,i,r)}const ew=b({logLoss_:tw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nw(e,t,n,s=ht.SUM_BY_NONZERO_WEIGHTS){const r=m(e,"labels","meanSquaredError"),a=m(t,"predictions","meanSquaredError");let o=null;n!=null&&(o=m(n,"weights","meanSquaredError")),lt(r.shape,a.shape,"Error in meanSquaredError: ");const i=So(r,a);return Mt(i,o,s)}const sw=b({meanSquaredError_:nw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rw(e,t){const n=m(e,"labels","sigmoidCrossEntropyWithLogits"),s=m(t,"logits","sigmoidCrossEntropyWithLogits");lt(n.shape,s.shape,"Error in sigmoidCrossEntropyWithLogits: ");const r=Xn(s),a=v(s,n),o=oo(be(Rt(Nt(s))));return F(P(r,a),o)}function aw(e,t,n,s=0,r=ht.SUM_BY_NONZERO_WEIGHTS){let a=m(e,"multiClassLabels","sigmoidCrossEntropy");const o=m(t,"logits","sigmoidCrossEntropy");let i=null;if(n!=null&&(i=m(n,"weights","sigmoidCrossEntropy")),lt(a.shape,o.shape,"Error in sigmoidCrossEntropy: "),s>0){const c=z(s),h=z(1),p=z(.5);a=F(v(a,P(h,c)),v(p,c))}const u=rw(a,o);return Mt(u,i,r)}const ow=b({sigmoidCrossEntropy_:aw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function iw(e,t,n=-1){if(n===-1&&(n=t.rank-1),n!==t.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${t.rank} and dim was ${n}`);return qt((r,a,o)=>{const u=uo(a,[n],!0),c=P(tt(a,"float32"),u);o([r,c]);const h=Rt(v(c,r));return{value:M(h,[n]),gradFunc:(d,y)=>{const[T,N]=y,S=qn(d.shape,[n]);return[v($(d,S),P(tt(T,"float32"),be(N))),v($(d,S),P(be(N),tt(T,"float32")))]}}})(e,t)}function uw(e,t,n,s=0,r=ht.SUM_BY_NONZERO_WEIGHTS){let a=m(e,"onehotLabels","softmaxCrossEntropy");const o=m(t,"logits","softmaxCrossEntropy");let i=null;if(n!=null&&(i=m(n,"weights","softmaxCrossEntropy")),lt(a.shape,o.shape,"Error in softmaxCrossEntropy: "),s>0){const c=z(s),h=z(1),p=z(a.shape[1]);a=F(v(a,P(h,c)),K(c,p))}const u=iw(a,o);return Mt(u,i,r)}const cw=b({softmaxCrossEntropy_:uw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lw(e,t,n,s){const r=m(e,"indices","sparseFillEmptyRows","int32"),a=m(t,"values","sparseFillEmptyRows"),o=m(n,"denseShape","sparseFillEmptyRows","int32"),i=m(s,"defaultValue","sparseFillEmptyRows",a.dtype);if(r.rank!==2)throw new Error(`Indices should be Tensor2D but received shape
        ${r.shape}`);if(a.rank!==1)throw new Error(`Values should be Tensor1D but received shape ${a.shape}`);if(o.rank!==1)throw new Error(`Dense shape should be Tensor1D but received shape ${o.shape}`);if(i.rank!==0)throw new Error(`Default value should be a scalar but received shape ${i.shape}`);const u={indices:r,values:a,denseShape:o,defaultValue:i},c=w.runKernel(jc,u);return{outputIndices:c[0],outputValues:c[1],emptyRowIndicator:c[2],reverseIndexMap:c[3]}}const hw=b({sparseFillEmptyRows_:lw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pw(e,t,n){const s=m(e,"inputIndices","sparseReshape","int32"),r=m(t,"inputShape","sparseReshape","int32"),a=m(n,"newShape","sparseReshape","int32");if(s.rank!==2)throw new Error(`Input indices should be Tensor2D but received shape
        ${s.shape}`);if(r.rank!==1)throw new Error(`Input shape should be Tensor1D but received shape ${r.shape}`);if(a.rank!==1)throw new Error(`New shape should be Tensor1D but received shape ${a.shape}`);const o={inputIndices:s,inputShape:r,newShape:a},i=w.runKernel(Wc,o);return{outputIndices:i[0],outputShape:i[1]}}const fw=b({sparseReshape_:pw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mw(e,t,n){const s=m(e,"data","sparseSegmentMean"),r=m(t,"indices","sparseSegmentMean","int32"),a=m(n,"segmentIds","sparseSegmentMean","int32");if(s.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
          ${r.shape}`);if(a.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
          ${a.shape}`);const o={data:s,indices:r,segmentIds:a};return w.runKernel(qc,o)}const dw=b({sparseSegmentMean_:mw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gw(e,t,n){const s=m(e,"data","sparseSegmentSum"),r=m(t,"indices","sparseSegmentSum","int32"),a=m(n,"segmentIds","sparseSegmentSum","int32");if(s.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
         ${r.shape}`);if(a.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
         ${a.shape}`);const o={data:s,indices:r,segmentIds:a};return w.runKernel(Gc,o)}const yw=b({sparseSegmentSum_:gw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bw(e,t,n,s,r,a,o,i){const u=m(e,"data","stringNGrams","string");if(u.dtype!=="string")throw new Error("Data must be of datatype string");if(u.shape.length!==1)throw new Error(`Data must be a vector, saw: ${u.shape}`);const c=m(t,"dataSplits","stringNGrams");if(c.dtype!=="int32")throw new Error("Data splits must be of datatype int32");const h={separator:n,nGramWidths:s,leftPad:r,rightPad:a,padWidth:o,preserveShortSequences:i},p={data:u,dataSplits:c},f=w.runKernel(Jc,p,h);return{nGrams:f[0],nGramsSplits:f[1]}}const ww=b({stringNGrams_:bw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nw(e,t,n=!0){const s=m(e,"input","stringSplit","string"),r=m(t,"delimiter","stringSplit","string");if(s.rank!==1)throw new Error(`Input should be Tensor1D but received shape ${s.shape}`);if(r.rank!==0)throw new Error(`Delimiter should be a scalar but received shape ${r.shape}`);const a={skipEmpty:n},o={input:s,delimiter:r},i=w.runKernel(Yc,o,a);return{indices:i[0],values:i[1],shape:i[2]}}const Tw=b({stringSplit_:Nw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sw(e,t){const n=m(e,"input","stringToHashBucketFast","string"),s={numBuckets:t};if(t<=0)throw new Error("Number of buckets must be at least 1");const r={input:n};return w.runKernel(Zc,r,s)}const $w=b({stringToHashBucketFast_:Sw});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kw(e,t,n,s=!0){const r=m(e,"input","staticRegexReplace","string"),a={pattern:t,rewrite:n,replaceGlobal:s};return w.runKernel(Hc,{x:r},a)}const Ew=b({staticRegexReplace_:kw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vw={fft:nr,ifft:Cn,rfft:sr,irfft:To},_w={hammingWindow:Hy,hannWindow:xo,frame:Io,stft:Zy},xw={flipLeftRight:nb,grayscaleToRGB:rb,resizeNearestNeighbor:Fb,resizeBilinear:Db,rgbToGrayscale:ob,rotateWithOffset:ub,cropAndResize:tb,nonMaxSuppression:lb,nonMaxSuppressionAsync:Tb,nonMaxSuppressionWithScore:$b,nonMaxSuppressionWithScoreAsync:Eb,nonMaxSuppressionPadded:_b,nonMaxSuppressionPaddedAsync:Ib,threshold:Rb,transform:Lb},Iw={bandPart:Vb,gramSchmidt:jb,qr:qb},Aw={absoluteDifference:Kb,computeWeightedLoss:Mt,cosineDistance:Xb,hingeLoss:Yb,huberLoss:Qb,logLoss:ew,meanSquaredError:sw,sigmoidCrossEntropy:ow,softmaxCrossEntropy:cw},Dw={sparseFillEmptyRows:hw,sparseReshape:fw,sparseSegmentMean:dw,sparseSegmentSum:yw},Ow={stringNGrams:ww,stringSplit:Tw,stringToHashBucketFast:$w,staticRegexReplace:Ew};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fw=new Map,Ts=new Map;class Ao{getClassName(){return this.constructor.className}static fromConfig(t,n){return new t(n)}}class Ht{constructor(){this.classNameMap={}}static getMap(){return Ht.instance==null&&(Ht.instance=new Ht),Ht.instance}static register(t){Ht.getMap().classNameMap[t.className]=[t,t.fromConfig]}}function Do(e,t,n){g(e.className!=null,()=>"Class being registered does not have the static className property defined."),g(typeof e.className=="string",()=>"className is required to be a string, but got type "+typeof e.className),g(e.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),typeof t>"u"&&(t="Custom"),typeof n>"u"&&(n=e.className);const s=n,r=t+">"+s;return Ht.register(e),Fw.set(r,e),Ts.set(e,r),e}function Cw(e){return Ts.has(e)?Ts.get(e):e.className}const MN=Object.freeze(Object.defineProperty({__proto__:null,Serializable:Ao,SerializationMap:Ht,getRegisteredName:Cw,registerClass:Do},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class _e extends Ao{minimize(t,n=!1,s){const{value:r,grads:a}=this.computeGradients(t,s);if(s!=null){const o=s.map(i=>({name:i.name,tensor:a[i.name]}));this.applyGradients(o)}else this.applyGradients(a);return pt(a),n?r:(r.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(t,n){return dm(t,n)}dispose(){this.iterations_!=null&&pt(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:z(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(t){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(t){return this.iterations_=(await t[0].tensor.data())[0],t.slice(1)}}Object.defineProperty(_e,Symbol.hasInstance,{value:e=>e.minimize!=null&&e.computeGradients!=null&&e.applyGradients!=null});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Bw extends _e{static get className(){return"Adadelta"}constructor(t,n,s=null){super(),this.learningRate=t,this.rho=n,this.epsilon=s,this.accumulatedGrads=[],this.accumulatedUpdates=[],s==null&&(this.epsilon=w.backend.epsilon())}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const a=w.registeredVariables[s],o=!1;this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${s}/accum_grad`,variable:V(()=>Tt(a).variable(o))}),this.accumulatedUpdates[r]==null&&(this.accumulatedUpdates[r]={originalName:`${s}/accum_var`,variable:V(()=>Tt(a).variable(o))});const i=Array.isArray(t)?t[r].tensor:t[s];if(i==null)return;const u=this.accumulatedGrads[r].variable,c=this.accumulatedUpdates[r].variable;V(()=>{const h=F(v(u,this.rho),v(At(i),1-this.rho)),p=v(K(Wt(F(c,this.epsilon)),Wt(F(u,this.epsilon))),i),f=F(v(c,this.rho),v(At(p),1-this.rho));u.assign(h),c.assign(f);const d=F(v(p,-this.learningRate),a);a.assign(d)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(pt(this.accumulatedGrads.map(t=>t.variable)),pt(this.accumulatedUpdates.map(t=>t.variable)))}async getWeights(){const t=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=t.length/2,s=!1;this.accumulatedGrads=t.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedUpdates=t.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.rho,n.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Rw extends _e{static get className(){return"Adagrad"}constructor(t,n=.1){super(),this.learningRate=t,this.initialAccumulatorValue=n,this.accumulatedGrads=[]}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const a=w.registeredVariables[s];this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${s}/accumulator`,variable:V(()=>dn(a.shape,this.initialAccumulatorValue).variable(!1))});const o=Array.isArray(t)?t[r].tensor:t[s];if(o==null)return;const i=this.accumulatedGrads[r].variable;V(()=>{const u=F(i,At(o));i.assign(u);const c=F(v(K(o,Wt(F(u,w.backend.epsilon()))),-this.learningRate),a);a.assign(c)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&pt(this.accumulatedGrads.map(t=>t.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulatedGrads=t.map(s=>({originalName:s.name,variable:s.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(t,n){return new t(n.learningRate,n.initialAccumulatorValue)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Pw extends _e{static get className(){return"Adam"}constructor(t,n,s,r=null){super(),this.learningRate=t,this.beta1=n,this.beta2=s,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],V(()=>{this.accBeta1=z(n).variable(),this.accBeta2=z(s).variable()}),r==null&&(this.epsilon=w.backend.epsilon())}applyGradients(t){const n=Array.isArray(t)?t.map(s=>s.name):Object.keys(t);V(()=>{const s=P(1,this.accBeta1),r=P(1,this.accBeta2);n.forEach((a,o)=>{const i=w.registeredVariables[a],u=!1;this.accumulatedFirstMoment[o]==null&&(this.accumulatedFirstMoment[o]={originalName:`${a}/m`,variable:V(()=>Tt(i).variable(u))}),this.accumulatedSecondMoment[o]==null&&(this.accumulatedSecondMoment[o]={originalName:`${a}/v`,variable:V(()=>Tt(i).variable(u))});const c=Array.isArray(t)?t[o].tensor:t[a];if(c==null)return;const h=this.accumulatedFirstMoment[o].variable,p=this.accumulatedSecondMoment[o].variable,f=F(v(h,this.beta1),v(c,1-this.beta1)),d=F(v(p,this.beta2),v(At(c),1-this.beta2)),y=K(f,s),T=K(d,r);h.assign(f),p.assign(d);const N=F(v(K(y,F(Wt(T),this.epsilon)),-this.learningRate),i);i.assign(N)}),this.accBeta1.assign(v(this.accBeta1,this.beta1)),this.accBeta2.assign(v(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&pt(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedSecondMoment!=null&&pt(this.accumulatedSecondMoment.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t),V(()=>{this.accBeta1.assign(rn(this.beta1,this.iterations_+1)),this.accBeta2.assign(rn(this.beta2,this.iterations_+1))});const n=t.length/2,s=!1;this.accumulatedFirstMoment=t.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedSecondMoment=t.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Lw extends _e{static get className(){return"Adamax"}constructor(t,n,s,r=null,a=0){super(),this.learningRate=t,this.beta1=n,this.beta2=s,this.epsilon=r,this.decay=a,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],V(()=>{this.iteration=z(0).variable(),this.accBeta1=z(n).variable()}),r==null&&(this.epsilon=w.backend.epsilon())}applyGradients(t){const n=Array.isArray(t)?t.map(s=>s.name):Object.keys(t);V(()=>{const s=P(1,this.accBeta1),r=K(-this.learningRate,F(v(this.iteration,this.decay),1));n.forEach((a,o)=>{const i=w.registeredVariables[a],u=!1;this.accumulatedFirstMoment[o]==null&&(this.accumulatedFirstMoment[o]={originalName:`${a}/m`,variable:Tt(i).variable(u)}),this.accumulatedWeightedInfNorm[o]==null&&(this.accumulatedWeightedInfNorm[o]={originalName:`${a}/v`,variable:Tt(i).variable(u)});const c=Array.isArray(t)?t[o].tensor:t[a];if(c==null)return;const h=this.accumulatedFirstMoment[o].variable,p=this.accumulatedWeightedInfNorm[o].variable,f=F(v(h,this.beta1),v(c,1-this.beta1)),d=v(p,this.beta2),y=Nt(c),T=po(d,y);h.assign(f),p.assign(T);const N=F(v(K(r,s),K(f,F(T,this.epsilon))),i);i.assign(N)}),this.iteration.assign(F(this.iteration,1)),this.accBeta1.assign(v(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&pt(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedWeightedInfNorm!=null&&pt(this.accumulatedWeightedInfNorm.map(t=>t.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(t){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon,n.decay)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Oo extends _e{static get className(){return"SGD"}constructor(t){super(),this.learningRate=t,this.setLearningRate(t)}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const a=Array.isArray(t)?t[r].tensor:t[s];if(a==null)return;const o=w.registeredVariables[s];V(()=>{const i=F(v(this.c,a),o);o.assign(i)})}),this.incrementIterations()}setLearningRate(t){this.learningRate=t,this.c!=null&&this.c.dispose(),this.c=Ct(z(-t))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(t){if(t=await this.extractIterations(t),t.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(t,n){return new t(n.learningRate)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class zw extends Oo{static get className(){return"Momentum"}constructor(t,n,s=!1){super(t),this.learningRate=t,this.momentum=n,this.useNesterov=s,this.accumulations=[],this.m=z(this.momentum)}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const a=w.registeredVariables[s];this.accumulations[r]==null&&(this.accumulations[r]={originalName:`${s}/momentum`,variable:V(()=>Tt(a).variable(!1))});const o=this.accumulations[r].variable,i=Array.isArray(t)?t[r].tensor:t[s];i!=null&&V(()=>{let u;const c=F(v(this.m,o),i);this.useNesterov?u=F(v(this.c,F(i,v(c,this.m))),a):u=F(v(this.c,c),a),o.assign(c),a.assign(u)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&pt(this.accumulations.map(t=>t.variable))}setMomentum(t){this.momentum=t}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulations=t.map(s=>({originalName:s.name,variable:s.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(t,n){return new t(n.learningRate,n.momentum,n.useNesterov)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Vw extends _e{static get className(){return"RMSProp"}constructor(t,n=.9,s=0,r=null,a=!1){if(super(),this.learningRate=t,this.decay=n,this.momentum=s,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=a,r==null&&(this.epsilon=w.backend.epsilon()),t==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const a=w.registeredVariables[s],o=!1;this.accumulatedMeanSquares[r]==null&&(this.accumulatedMeanSquares[r]={originalName:`${s}/rms`,variable:V(()=>Tt(a).variable(o))}),this.accumulatedMoments[r]==null&&(this.accumulatedMoments[r]={originalName:`${s}/momentum`,variable:V(()=>Tt(a).variable(o))}),this.accumulatedMeanGrads[r]==null&&this.centered&&(this.accumulatedMeanGrads[r]={originalName:`${s}/mg`,variable:V(()=>Tt(a).variable(o))});const i=Array.isArray(t)?t[r].tensor:t[s];if(i==null)return;const u=this.accumulatedMeanSquares[r].variable,c=this.accumulatedMoments[r].variable;V(()=>{const h=F(v(u,this.decay),v(At(i),1-this.decay));if(this.centered){const p=this.accumulatedMeanGrads[r].variable,f=F(v(p,this.decay),v(i,1-this.decay)),d=K(v(i,this.learningRate),Wt(P(h,F(At(f),this.epsilon)))),y=F(v(c,this.momentum),d);u.assign(h),p.assign(f),c.assign(y);const T=P(a,y);a.assign(T)}else{const p=F(v(u,this.decay),v(At(i),1-this.decay)),f=F(v(c,this.momentum),K(v(i,this.learningRate),Wt(F(p,this.epsilon))));u.assign(p),c.assign(f);const d=P(a,f);a.assign(d)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&pt(this.accumulatedMeanSquares.map(t=>t.variable)),this.accumulatedMeanGrads!=null&&this.centered&&pt(this.accumulatedMeanGrads.map(t=>t.variable)),this.accumulatedMoments!=null&&pt(this.accumulatedMoments.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&t.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=this.centered?t.length/3:t.length/2,s=!1;this.accumulatedMeanSquares=t.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedMoments=t.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.centered&&(this.accumulatedMeanGrads=t.slice(n*2,n*3).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(t,n){return new t(n.learningRate,n.decay,n.momentum,n.epsilon,n.centered)}}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Uw=[Bw,Rw,Pw,Lw,zw,Vw,Oo];function jw(){for(const e of Uw)Do(e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ww="model",qw=".json",Gw=".weights.bin";function Pr(e){return new Promise(t=>setTimeout(t)).then(e)}class Ne{constructor(t){if(!C().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");t.startsWith(Ne.URL_SCHEME)&&(t=t.slice(Ne.URL_SCHEME.length)),(t==null||t.length===0)&&(t=Ww),this.modelJsonFileName=t+qw,this.weightDataFileName=t+Gw}async save(t){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const n=Ot.join(t.weightData),s=window.URL.createObjectURL(new Blob([n],{type:"application/octet-stream"}));if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const r=[{paths:["./"+this.weightDataFileName],weights:t.weightSpecs}],a=Ba(t,r),o=window.URL.createObjectURL(new Blob([JSON.stringify(a)],{type:"application/json"})),i=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(i.download=this.modelJsonFileName,i.href=o,await Pr(()=>i.dispatchEvent(new MouseEvent("click"))),t.weightData!=null){const u=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;u.download=this.weightDataFileName,u.href=s,await Pr(()=>u.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:mn(t)}}}}Ne.URL_SCHEME="downloads://";class Mw{constructor(t){if(t==null||t.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${t}`);this.jsonFile=t[0],this.weightsFiles=t.slice(1)}async load(){return new Promise((t,n)=>{const s=new FileReader;s.onload=r=>{const a=JSON.parse(r.target.result),o=a.modelTopology;if(o==null){n(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(a.weightsManifest==null){n(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){t({modelTopology:o});return}const u=Ws(a,c=>this.loadWeights(c));t(u)},s.onerror=r=>n(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),s.readAsText(this.jsonFile)})}loadWeights(t){const n=[],s=[];for(const o of t)n.push(...o.weights),s.push(...o.paths);const r=this.checkManifestAndWeightFiles(t),a=s.map(o=>this.loadWeightsFile(o,r[o]));return Promise.all(a).then(o=>[n,o])}loadWeightsFile(t,n){return new Promise((s,r)=>{const a=new FileReader;a.onload=o=>{const i=o.target.result;s(i)},a.onerror=o=>r(`Failed to weights data from file of path '${t}'.`),a.readAsArrayBuffer(n)})}checkManifestAndWeightFiles(t){const n=[],s=this.weightsFiles.map(a=>Cr(a.name)),r={};for(const a of t)a.paths.forEach(o=>{const i=Cr(o);if(n.indexOf(i)!==-1)throw new Error(`Duplicate file basename found in weights manifest: '${i}'`);if(n.push(i),s.indexOf(i)===-1)throw new Error(`Weight file with basename '${i}' is not provided.`);r[o]=this.weightsFiles[s.indexOf(i)]});if(n.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${n.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}}const Kw=e=>C().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Ne.URL_SCHEME)?Hw(e.slice(Ne.URL_SCHEME.length)):null;Y.registerSaveRouter(Kw);function Hw(e="model"){return new Ne(e)}function Xw(e){return new Mw(e)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lr(e,t,n,s){o(e),n=n??0,s=s??1,i(n,s);let r=0;const a=u=>(u.then(c=>{const h=n+ ++r/e.length*(s-n);return t(h),c}),u);function o(u){g(u!=null&&Array.isArray(u)&&u.length>0,()=>"promises must be a none empty array")}function i(u,c){g(u>=0&&u<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${u}`),g(c>=0&&c<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${c}`),g(c>=u,()=>`startFraction must be no more than endFraction, but got startFraction ${u} and endFraction ${c}`)}return Promise.all(e.map(a))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Fo(e,t){t==null&&(t={});const n=t.fetchFunc==null?C().platform.fetch:t.fetchFunc,s=e.map(p=>n(p,t.requestInit,{isBinary:!0})),i=(t.onProgress==null?await Promise.all(s):await Lr(s,t.onProgress,0,.5)).map(p=>p.arrayBuffer());return t.onProgress==null?await Promise.all(i):await Lr(i,t.onProgress,.5,1)}function Jw(e,t){var n;const s=t.fetchFunc==null?C().platform.fetch:t.fetchFunc;let r=0,a;return(n=t.onProgress)===null||n===void 0||n.call(t,0),new ReadableStream({pull:async o=>{for(var i;r<e.length;){a||(a=(await s(e[r],t.requestInit,{isBinary:!0})).body.getReader());const{done:u,value:c}=await a.read();if(u){r++,a=void 0,(i=t.onProgress)===null||i===void 0||i.call(t,r/e.length);continue}o.enqueue(c);return}o.close()}})}async function Yw(e,t="",n,s){return Co(o=>Fo(o,{requestInit:s}))(e,t,n)}function Co(e){return async(t,n="",s)=>{const r=t.map(()=>!1),a={},o=s!=null?s.map(()=>!1):[],i=[];if(t.forEach((d,y)=>{let T=0;d.weights.forEach(N=>{const S="quantization"in N?N.quantization.dtype:N.dtype,I=de[S]*Z(N.shape),A=()=>{r[y]=!0,a[y]==null&&(a[y]=[]),a[y].push({manifestEntry:N,groupOffset:T,sizeBytes:I})};s!=null?s.forEach((k,_)=>{k===N.name&&(A(),o[_]=!0)}):A(),i.push(N.name),T+=I})}),!o.every(d=>d)){const d=s.filter((y,T)=>!o[T]);throw new Error(`Could not find weights in manifest with names: ${d.join(", ")}. 
Manifest JSON has weights with names: ${i.join(", ")}.`)}const u=r.reduce((d,y,T)=>(y&&d.push(T),d),[]),c=[];u.forEach(d=>{t[d].paths.forEach(y=>{const T=n+(n.endsWith("/")?"":"/")+y;c.push(T)})});const h=await e(c),p={};let f=0;return u.forEach(d=>{const y=t[d].paths.length,T=new Ot(h.slice(f,f+y));a[d].forEach(S=>{const I=T.slice(S.groupOffset,S.groupOffset+S.sizeBytes),A=Oa(I,[S.manifestEntry]);for(const k in A)p[k]=A[k]}),f+=y}),p}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zw="application/octet-stream",Qw="application/json";class pr{constructor(t,n){if(this.DEFAULT_METHOD="POST",n==null&&(n={}),this.weightPathPrefix=n.weightPathPrefix,this.weightUrlConverter=n.weightUrlConverter,n.fetchFunc!=null?(g(typeof n.fetchFunc=="function",()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=n.fetchFunc):this.fetch=C().platform.fetch,g(t!=null&&t.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(t)&&g(t.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${t.length}).`),this.path=t,n.requestInit!=null&&n.requestInit.body!=null)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=n.requestInit||{},this.loadOptions=n}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const n=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);n.body=new FormData;const s=[{paths:["./model.weights.bin"],weights:t.weightSpecs}],r=Ba(t,s);if(n.body.append("model.json",new Blob([JSON.stringify(r)],{type:Qw}),"model.json"),t.weightData!=null){const o=Ot.join(t.weightData);n.body.append("model.weights.bin",new Blob([o],{type:Zw}),"model.weights.bin")}const a=await this.fetch(this.path,n);if(a.ok)return{modelArtifactsInfo:mn(t),responses:[a]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${a.status}.`)}async loadModelJSON(){const t=await this.fetch(this.path,this.requestInit);if(!t.ok)throw new Error(`Request to ${this.path} failed with status code ${t.status}. Please verify this URL points to the model JSON of the model to load.`);let n;try{n=await t.json()}catch{let o=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?o+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":o+=" Please make sure the server is serving valid JSON for this request.",new Error(o)}const s=n.modelTopology,r=n.weightsManifest;if(s==null&&r==null)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return n}async load(){if(this.loadOptions.streamWeights)return this.loadStream();const t=await this.loadModelJSON();return Ws(t,n=>this.loadWeights(n))}async loadStream(){const t=await this.loadModelJSON(),n=await this.getWeightUrls(t.weightsManifest),s=In(t.weightsManifest),r=()=>Jw(n,this.loadOptions);return Object.assign(Object.assign({},t),{weightSpecs:s,getWeightStream:r})}async getWeightUrls(t){const n=Array.isArray(this.path)?this.path[1]:this.path,[s,r]=t0(n),a=this.weightPathPrefix||s,o=[],i=[];for(const u of t)for(const c of u.paths)this.weightUrlConverter!=null?i.push(this.weightUrlConverter(c)):o.push(a+c+r);return this.weightUrlConverter&&o.push(...await Promise.all(i)),o}async loadWeights(t){const n=await this.getWeightUrls(t),s=In(t),r=await Fo(n,this.loadOptions);return[s,r]}}pr.URL_SCHEME_REGEX=/^https?:\/\//;function t0(e){const t=e.lastIndexOf("/"),n=e.lastIndexOf("?"),s=e.substring(0,t),r=n>t?e.substring(n):"";return[s+"/",r]}function Ss(e){return e.match(pr.URL_SCHEME_REGEX)!=null}const Bo=(e,t)=>{if(typeof fetch>"u"&&(t==null||t.fetchFunc==null))return null;{let n=!0;if(Array.isArray(e)?n=e.every(s=>Ss(s)):n=Ss(e),n)return fr(e,t)}return null};Y.registerSaveRouter(Bo);Y.registerLoadRouter(Bo);function fr(e,t){return new pr(e,t)}function e0(e,t){return fr(e,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class es{constructor(t){this.modelArtifacts=t}load(){return this.modelArtifacts}}class Ro{constructor(t){this.saveHandler=t}save(t){return this.saveHandler(t)}}class n0{constructor(t){t.load&&(this.load=()=>Promise.resolve(t.load())),t.save&&(this.save=n=>Promise.resolve(t.save(n)))}}function s0(e,t,n,s){const r=arguments;return new n0(Bn(...r))}function Bn(e,t,n,s){return arguments.length===1?e.modelTopology!=null||e.weightSpecs!=null?new es(e):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new es({modelTopology:e})):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new es({modelTopology:e,weightSpecs:t,weightData:n,trainingConfig:s}))}function r0(e){return new Ro(e)}function a0(e){return new Ro(e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Po=Object.freeze(Object.defineProperty({__proto__:null,CompositeArrayBuffer:Ot,browserFiles:Xw,browserHTTPRequest:e0,concatenateArrayBuffers:Hl,copyModel:gh,decodeWeights:Oa,decodeWeightsStream:Ca,encodeWeights:jl,fromMemory:s0,fromMemorySync:Bn,getLoadHandlers:nh,getModelArtifactsForJSON:Ws,getModelArtifactsForJSONSync:js,getModelArtifactsInfoForJSON:mn,getSaveHandlers:eh,getWeightSpecs:In,http:fr,isHTTPScheme:Ss,listModels:mh,loadWeights:Yw,moveModel:yh,registerLoadRouter:th,registerSaveRouter:Ql,removeModel:dh,weightsLoaderFactory:Co,withSaveHandler:r0,withSaveHandlerSync:a0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let oe,zr=!1;function Lo(e,t=3){if(t>4)throw new Error("Cannot construct Tensor with more than 4 channels from pixels.");if(e==null)throw new Error("pixels passed to tf.browser.fromPixels() can not be null");let n=!1,s=!1,r=!1,a=!1,o=!1,i=!1;if(e.data instanceof Uint8Array)n=!0;else if(typeof ImageData<"u"&&e instanceof ImageData)s=!0;else if(typeof HTMLVideoElement<"u"&&e instanceof HTMLVideoElement)r=!0;else if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement)a=!0;else if(e.getContext!=null)o=!0;else if(typeof ImageBitmap<"u"&&e instanceof ImageBitmap)i=!0;else throw new Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${e.constructor.name}`);if(vn(yr,w.backendName)!=null){const y={pixels:e},T={numChannels:t};return w.runKernel(yr,y,T)}const[c,h]=r?[e.videoWidth,e.videoHeight]:[e.width,e.height];let p;if(o)p=e.getContext("2d").getImageData(0,0,c,h).data;else if(s||n)p=e.data;else if(a||r||i){if(oe==null)if(typeof document>"u")if(typeof OffscreenCanvas<"u"&&typeof OffscreenCanvasRenderingContext2D<"u")oe=new OffscreenCanvas(1,1).getContext("2d");else throw new Error("Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.");else oe=document.createElement("canvas").getContext("2d",{willReadFrequently:!0});oe.canvas.width=c,oe.canvas.height=h,oe.drawImage(e,0,0,c,h),p=oe.getImageData(0,0,c,h).data}let f;if(t===4)f=new Int32Array(p);else{const y=c*h;f=new Int32Array(y*t);for(let T=0;T<y;T++)for(let N=0;N<t;++N)f[T*t+N]=p[T*4+N]}return ko(f,[h,c,t],"int32")}function o0(e){return e!=null&&e.data instanceof Uint8Array}function i0(){return typeof window<"u"&&typeof ImageBitmap<"u"&&window.hasOwnProperty("createImageBitmap")}function u0(e){return e!=null&&e.width!==0&&e.height!==0}function c0(e){return i0()&&!(e instanceof ImageBitmap)&&u0(e)&&!o0(e)}async function l0(e,t=3){let n=null;if(C().getBool("WRAP_TO_IMAGEBITMAP")&&c0(e)){let s;try{s=await createImageBitmap(e,{premultiplyAlpha:"none"})}catch{s=null}s!=null&&s.width===e.width&&s.height===e.height?n=s:n=e}else n=e;return Lo(n,t)}function zo(e){if(e.rank!==2&&e.rank!==3)throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${e.rank}.`);const t=e.rank===2?1:e.shape[2];if(t>4||t===2)throw new Error(`toPixels only supports depth of size 1, 3 or 4 but got ${t}`);if(e.dtype!=="float32"&&e.dtype!=="int32")throw new Error(`Unsupported type for toPixels: ${e.dtype}. Please use float32 or int32 tensors.`)}function h0(e){const t=(e==null?void 0:e.alpha)||1;if(t>1||t<0)throw new Error(`Alpha value ${t} is suppoed to be in range [0 - 1].`)}async function p0(e,t){let n=m(e,"img","toPixels");if(!(e instanceof et)){const c=n;n=tt(c,"int32"),c.dispose()}zo(n);const[s,r]=n.shape.slice(0,2),a=n.rank===2?1:n.shape[2],o=await n.data(),i=n.dtype==="float32"?255:1,u=new Uint8ClampedArray(r*s*4);for(let c=0;c<s*r;++c){const h=[0,0,0,255];for(let f=0;f<a;f++){const d=o[c*a+f];if(n.dtype==="float32"){if(d<0||d>1)throw new Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${d}.`)}else if(n.dtype==="int32"&&(d<0||d>255))throw new Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${d}.`);a===1?(h[0]=d*i,h[1]=d*i,h[2]=d*i):h[f]=d*i}const p=c*4;u[p+0]=Math.round(h[0]),u[p+1]=Math.round(h[1]),u[p+2]=Math.round(h[2]),u[p+3]=Math.round(h[3])}if(t!=null){zr||vn(ca,w.backendName)!=null&&(console.warn("tf.browser.toPixels is not efficient to draw tensor on canvas. Please try tf.browser.draw instead."),zr=!0),t.width=r,t.height=s;const c=t.getContext("2d"),h=new ImageData(u,r,s);c.putImageData(h,0,0)}return n!==e&&n.dispose(),u}function f0(e,t,n){let s=m(e,"img","draw");if(!(e instanceof et)){const o=s;s=tt(o,"int32"),o.dispose()}zo(s),h0(n==null?void 0:n.imageOptions);const r={image:s},a={canvas:t,options:n};w.runKernel(ca,r,a)}const m0=b({fromPixels_:Lo}),KN=Object.freeze(Object.defineProperty({__proto__:null,draw:f0,fromPixels:m0,fromPixelsAsync:l0,toPixels:p0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */jw();/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const d0=C();d0.registerFlag("KEEP_INTERMEDIATE_TENSORS",()=>!1,e=>{e&&console.warn("Keep intermediate tensors is ON. This will print the values of all intermediate tensors during model inference. Not all models support this mode. For details, check e2e/benchmarks/ model_config.js. This significantly impacts performance.")});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */var dt;(function(e){e[e.DT_INVALID=0]="DT_INVALID",e[e.DT_FLOAT=1]="DT_FLOAT",e[e.DT_DOUBLE=2]="DT_DOUBLE",e[e.DT_INT32=3]="DT_INT32",e[e.DT_UINT8=4]="DT_UINT8",e[e.DT_INT16=5]="DT_INT16",e[e.DT_INT8=6]="DT_INT8",e[e.DT_STRING=7]="DT_STRING",e[e.DT_COMPLEX64=8]="DT_COMPLEX64",e[e.DT_INT64=9]="DT_INT64",e[e.DT_BOOL=10]="DT_BOOL",e[e.DT_QINT8=11]="DT_QINT8",e[e.DT_QUINT8=12]="DT_QUINT8",e[e.DT_QINT32=13]="DT_QINT32",e[e.DT_BFLOAT16=14]="DT_BFLOAT16",e[e.DT_QINT16=15]="DT_QINT16",e[e.DT_QUINT16=16]="DT_QUINT16",e[e.DT_UINT16=17]="DT_UINT16",e[e.DT_COMPLEX128=18]="DT_COMPLEX128",e[e.DT_HALF=19]="DT_HALF",e[e.DT_RESOURCE=20]="DT_RESOURCE",e[e.DT_VARIANT=21]="DT_VARIANT",e[e.DT_UINT32=22]="DT_UINT32",e[e.DT_UINT64=23]="DT_UINT64",e[e.DT_FLOAT_REF=101]="DT_FLOAT_REF",e[e.DT_DOUBLE_REF=102]="DT_DOUBLE_REF",e[e.DT_INT32_REF=103]="DT_INT32_REF",e[e.DT_UINT8_REF=104]="DT_UINT8_REF",e[e.DT_INT16_REF=105]="DT_INT16_REF",e[e.DT_INT8_REF=106]="DT_INT8_REF",e[e.DT_STRING_REF=107]="DT_STRING_REF",e[e.DT_COMPLEX64_REF=108]="DT_COMPLEX64_REF",e[e.DT_INT64_REF=109]="DT_INT64_REF",e[e.DT_BOOL_REF=110]="DT_BOOL_REF",e[e.DT_QINT8_REF=111]="DT_QINT8_REF",e[e.DT_QUINT8_REF=112]="DT_QUINT8_REF",e[e.DT_QINT32_REF=113]="DT_QINT32_REF",e[e.DT_BFLOAT16_REF=114]="DT_BFLOAT16_REF",e[e.DT_QINT16_REF=115]="DT_QINT16_REF",e[e.DT_QUINT16_REF=116]="DT_QUINT16_REF",e[e.DT_UINT16_REF=117]="DT_UINT16_REF",e[e.DT_COMPLEX128_REF=118]="DT_COMPLEX128_REF",e[e.DT_HALF_REF=119]="DT_HALF_REF",e[e.DT_RESOURCE_REF=120]="DT_RESOURCE_REF",e[e.DT_VARIANT_REF=121]="DT_VARIANT_REF",e[e.DT_UINT32_REF=122]="DT_UINT32_REF",e[e.DT_UINT64_REF=123]="DT_UINT64_REF"})(dt||(dt={}));var Vr;(function(e){(function(t){t[t.LEGACY=0]="LEGACY",t[t.V1=1]="V1",t[t.V2=2]="V2"})(e.CheckpointFormatVersion||(e.CheckpointFormatVersion={}))})(Vr||(Vr={}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mr={};function HN(e,t){const n={tfOpName:e,category:"custom",inputs:[],attrs:[],customExecutor:t};mr[e]=n}function Vo(e){return mr[e]}function XN(e){delete mr[e]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function l(e,t,n,s,r){const a=t.inputParams[e];if(a&&a.inputIndexStart!==void 0){const i=a.inputIndexStart,u=a.inputIndexEnd===0?void 0:a.inputIndexEnd===void 0?i+1:a.inputIndexEnd,c=i<0?t.inputNames.length+i:i;if(a.type==="tensor")return at(t.inputNames[c],n,s,r);if(a.type==="tensors"){const f=t.inputs.slice(i,u);return t.inputNames.slice(i,u).filter((y,T)=>{var N;return((N=f[T])===null||N===void 0?void 0:N.op)!=="NoOp"}).map(y=>at(y,n,s,r))}const h=at(t.inputNames[c],n,s,r),p=h.dataSync();return a.type==="number"?p[0]:me(h.shape,p)}const o=t.attrParams[e];return o&&o.value}function at(e,t,n,s){const[r,a]=gt(e,n);if(s!=null){const i=s.getHashTableHandleByName(r);if(i!=null)return i}const o=n.currentContextIds.find(i=>!!t[Rn(r,i)]);return o!==void 0?t[Rn(r,o)][a]:void 0}function Ur(e,t,n){return t[Rn(e,n.currentContextId)]}function zt(e,t){const[n,s,r]=gt(e,t);return[Rn(n,t&&t.currentContextId),s,r]}function Rn(e,t){return t?`${e}-${t}`:e}function gt(e,t){if(e==="")return["",0,void 0];const n=t!=null&&t.parseNodeNameCache!=null;if(n){const a=t.parseNodeNameCache.get(e);if(a!=null)return a}const s=e.split(":");let r;if(s.length===1)r=[e,0,void 0];else{const a=s[0],o=s.length===3?s[1]:void 0,i=Number(s[s.length-1]);r=[a,i,o]}return n&&t.parseNodeNameCache.set(e,r),r}function $n(e,t,n){let s=l("pad",e,t,n);if(s==="explicit"){s=l("explicitPaddings",e,t,n);const r=[[0,0],[0,0],[0,0],[0,0]];for(let a=0;a<4;a++)r[a][0]=s[a*2],r[a][1]=s[a*2+1];return r}return s}function Vt(e){return e.kept?e:Yt(e)}/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const g0=[{tfOpName:"Add",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddV2",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddN",category:"arithmetic",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"BiasAdd",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"Sub",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"RealDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Div",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"DivNoNan",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mul",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Maximum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Minimum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Pow",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SquaredDifference",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorMod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],y0=Object.freeze(Object.defineProperty({__proto__:null,json:g0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const b0=[{tfOpName:"Abs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan2",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Ceil",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ClipByValue",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"clipValueMin",type:"number"},{start:2,name:"clipValueMax",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Complex",category:"basic_math",inputs:[{start:0,name:"real",type:"tensor"},{start:1,name:"imag",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ComplexAbs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Elu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Exp",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Floor",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Imag",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Neg",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Real",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Prelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"alpha",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu6",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Selu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sigmoid",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Rsqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Square",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sign",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Round",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Expm1",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log1p",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Reciprocal",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Softplus",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Erf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LeakyRelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"alpha",name:"alpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsNan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsFinite",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsInf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],w0=Object.freeze(Object.defineProperty({__proto__:null,json:b0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const N0=[{tfOpName:"EmptyTensorList",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"maxNumElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"LoopCond",category:"control",inputs:[{start:0,name:"pred",type:"tensor"}]},{tfOpName:"Switch",category:"control",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"pred",type:"tensor"}]},{tfOpName:"Merge",category:"control",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"Enter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"frame_name",name:"frameName",type:"string"},{tfName:"is_constant",name:"isConstant",type:"bool"}]},{tfOpName:"Exit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NextIteration",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayV3",category:"control",inputs:[{start:0,name:"size",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"dynamic_size",name:"dynamicSize",type:"bool"},{tfName:"clear_after_read",name:"clearAfterRead",type:"bool"},{tfName:"identical_element_shapes",name:"identicalElementShapes",type:"bool"},{tfName:"tensor_array_name",name:"name",type:"string"}]},{tfOpName:"TensorArrayWriteV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayReadV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayGatherV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"}]},{tfOpName:"TensorArrayScatterV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArrayConcatV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape_except0",name:"elementShapeExcept0",type:"shape",notSupported:!0}]},{tfOpName:"TensorArraySplitV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"tensor",type:"tensor"},{start:2,name:"lengths",type:"number[]"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArraySizeV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}]},{tfOpName:"TensorArrayCloseV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"}]},{tfOpName:"StatelessIf",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"If",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"StatelessWhile",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"While",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"TensorListScatter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListScatterV2",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"},{start:3,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGather",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListSetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListReserve",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListFromTensor",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListStack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"},{tfName:"num_elements",name:"numElements",type:"dtype"}]},{tfOpName:"TensorListSplit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"},{start:2,name:"lengths",type:"number[]"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcat",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcatV2",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPopBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPushBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListLength",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}]},{tfOpName:"TensorListResize",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"size",type:"number"}]}],T0=Object.freeze(Object.defineProperty({__proto__:null,json:N0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const S0=[{tfOpName:"AvgPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[],notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPoolWithArgmax",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"include_batch_in_index",name:"includeBatchInIndex",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AvgPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Conv1D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"stride",name:"stride",type:"number"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NWC"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"dilation",name:"dilation",type:"number",defaultValue:1}]},{tfOpName:"Conv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"useCudnnOnGpu",name:"useCudnnOnGpu",type:"bool"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"_FusedConv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"use_cudnn_on_gpu",name:"useCudnnOnGpu",type:"bool",defaultValue:!0},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2}]},{tfOpName:"Conv2DBackpropInput",category:"convolution",inputs:[{start:2,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:0,name:"outputShape",type:"number[]"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]",notSupported:!0}]},{tfOpName:"DepthwiseConv2d",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"DepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"FusedDepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]}]},{tfOpName:"Conv3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"Dilation2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"rates",name:"dilations",type:"number[]"},{tfName:"padding",name:"pad",type:"string"}]}],$0=Object.freeze(Object.defineProperty({__proto__:null,json:S0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const k0=[{tfOpName:"Fill",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"},{start:1,name:"value",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"LinSpace",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"num",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"OneHot",category:"creation",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"depth",type:"number"},{start:2,name:"onValue",type:"number",defaultValue:1},{start:3,name:"offValue",type:"number",defaultValue:0}],attrs:[{tfName:"axis",name:"axis",type:"number",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Ones",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"OnesLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"RandomStandardNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniform",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number",defaultValue:0},{tfName:"maxval",name:"maxval",type:"number",defaultValue:1},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniformInt",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number"},{tfName:"maxval",name:"maxval",type:"number"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Range",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"step",type:"number",defaultValue:0}],attrs:[{tfName:"Tidx",name:"dtype",type:"dtype"}]},{tfOpName:"TruncatedNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"means",name:"mean",type:"number",defaultValue:0},{tfName:"stddev",name:"stdDev",type:"number",defaultValue:1},{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Zeros",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"ZerosLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Multinomial",category:"creation",inputs:[{start:0,name:"logits",type:"tensor"},{start:1,name:"numSamples",type:"number"}],attrs:[{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number"},{tfName:"T",name:"dtype",type:"dtype"},{tfName:"output_dtype",name:"output_dtype",type:"dtype"}]}],E0=Object.freeze(Object.defineProperty({__proto__:null,json:k0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const v0=[{tfOpName:"NonMaxSuppressionV2",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV3",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV4",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"T_threshold",name:"threshold",type:"dtype",notSupported:!0},{tfName:"pad_to_max_output_size",name:"padToMaxOutputSize",type:"bool"}]},{tfOpName:"NonMaxSuppressionV5",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"},{start:5,name:"softNmsSigma",type:"number"}]},{tfOpName:"Where",category:"dynamic",inputs:[{start:0,name:"condition",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ListDiff",category:"dynamic",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],_0=Object.freeze(Object.defineProperty({__proto__:null,json:v0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const x0=[{tfOpName:"LowerBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"TopKV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"k",type:"number"}],attrs:[{tfName:"sorted",name:"sorted",type:"bool"}]},{tfOpName:"UpperBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"Unique",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"UniqueV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]}],I0=Object.freeze(Object.defineProperty({__proto__:null,json:x0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const A0=[{tfOpName:"PlaceholderWithDefault",category:"graph",inputs:[{start:0,name:"default",type:"tensor"}],attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Placeholder",category:"graph",attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Const",category:"graph"},{tfOpName:"Identity",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IdentityN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Snapshot",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Rank",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Size",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Shape",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"ShapeN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Print",category:"graph",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"data",type:"tensors"}],attrs:[{tfName:"message",name:"message",type:"string"},{tfName:"first_n",name:"firstN",type:"number",notSupported:!0},{tfName:"summarize",name:"summarize",type:"number",defaultValue:3}]},{tfOpName:"NoOp",category:"graph",inputs:[]},{tfOpName:"StopGradient",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"FakeQuantWithMinMaxVars",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"min",name:"min",type:"number"},{tfName:"max",name:"max",type:"number"}]}],D0=Object.freeze(Object.defineProperty({__proto__:null,json:A0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const O0=[{tfOpName:"HashTable",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"HashTableV2",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"LookupTableImport",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableImportV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFind",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFindV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableSize",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"LookupTableSizeV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"InitializeTable",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]},{tfOpName:"InitializeTableV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],F0=Object.freeze(Object.defineProperty({__proto__:null,json:O0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const C0=[{tfOpName:"ResizeBilinear",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ResizeNearestNeighbor",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"CropAndResize",category:"image",inputs:[{start:0,name:"image",type:"tensor"},{start:1,name:"boxes",type:"tensor"},{start:2,name:"boxInd",type:"tensor"},{start:3,name:"cropSize",type:"number[]"}],attrs:[{tfName:"method",name:"method",type:"string"},{tfName:"extrapolation_value",name:"extrapolationValue",type:"number"}]},{tfOpName:"ImageProjectiveTransformV3",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"transforms",type:"tensor"},{start:2,name:"outputShape",type:"number[]"},{start:3,name:"fillValue",type:"number"}],attrs:[{tfName:"interpolation",name:"interpolation",type:"string"},{tfName:"fill_mode",name:"fillMode",type:"string"}]}],B0=Object.freeze(Object.defineProperty({__proto__:null,json:C0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const R0=[{tfOpName:"Equal",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NotEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Greater",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"GreaterEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Less",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LessEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalAnd",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalNot",category:"logical",inputs:[{start:0,name:"a",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalOr",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Select",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SelectV2",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BitwiseAnd",category:"logical",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}]}],P0=Object.freeze(Object.defineProperty({__proto__:null,json:R0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const L0=[{tfOpName:"_FusedMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMulV2",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Transpose",category:"matrices",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"perm",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Einsum",category:"matrices",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"equation",name:"equation",type:"string"},{tfName:"N",name:"n",type:"number",defaultValue:2},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"MatrixBandPart",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"numLower",type:"tensor"},{start:1,name:"numUpper",type:"tensor"}]}],z0=Object.freeze(Object.defineProperty({__proto__:null,json:L0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const V0=[{tfOpName:"EuclideanNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",defaultValue:!1}]},{tfOpName:"FusedBatchNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV2",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV3",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"LRN",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"depth_radius",name:"radius",type:"number",defaultValue:5},{tfName:"bias",name:"bias",type:"number",defaultValue:1},{tfName:"alpha",name:"alpha",type:"number",defaultValue:1},{tfName:"beta",name:"beta",type:"number",defaultValue:.5}]},{tfOpName:"Softmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"LogSoftmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]}],U0=Object.freeze(Object.defineProperty({__proto__:null,json:V0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const j0=[{tfOpName:"Bincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}]},{tfOpName:"DenseBincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}],attrs:[{tfName:"binary_output",name:"binaryOutput",type:"bool"}]},{tfOpName:"Max",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Mean",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Min",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Sum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"All",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Any",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"ArgMax",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"ArgMin",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"Prod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cumprod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]},{tfOpName:"Cumsum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]}],W0=Object.freeze(Object.defineProperty({__proto__:null,json:j0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const q0=[{tfOpName:"ConcatV2",category:"slice_join",inputs:[{start:0,end:-1,name:"tensors",type:"tensors"},{start:-1,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"Concat",category:"slice_join",inputs:[{start:1,end:0,name:"tensors",type:"tensors"},{start:0,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"GatherV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"axis",type:"number",defaultValue:0}],attrs:[{tfName:"batch_dims",name:"batchDims",type:"number",defaultValue:0}]},{tfOpName:"Gather",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",notSupported:!0}]},{tfOpName:"Reverse",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"dims",type:"bool[]"}]},{tfOpName:"ReverseV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}]},{tfOpName:"Slice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"size",type:"number[]"}]},{tfOpName:"StridedSlice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"end",type:"number[]"},{start:3,name:"strides",type:"number[]"}],attrs:[{tfName:"begin_mask",name:"beginMask",type:"number",defaultValue:0},{tfName:"end_mask",name:"endMask",type:"number",defaultValue:0},{tfName:"new_axis_mask",name:"newAxisMask",type:"number",defaultValue:0},{tfName:"ellipsis_mask",name:"ellipsisMask",type:"number",defaultValue:0},{tfName:"shrink_axis_mask",name:"shrinkAxisMask",type:"number",defaultValue:0}]},{tfOpName:"Pack",category:"slice_join",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0}]},{tfOpName:"Unpack",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0},{tfName:"num",name:"num",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Tile",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"reps",type:"number[]"}]},{tfOpName:"Split",category:"slice_join",inputs:[{start:0,name:"axis",type:"number",defaultValue:0},{start:1,name:"x",type:"tensor"}],attrs:[{tfName:"num_split",name:"numOrSizeSplits",type:"number",defaultValue:1}]},{tfOpName:"SplitV",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"numOrSizeSplits",type:"number[]"},{start:2,name:"axis",type:"number",defaultValue:0}]},{tfOpName:"ScatterNd",category:"slice_join",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"shape",type:"number[]"}]},{tfOpName:"GatherNd",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}]},{tfOpName:"SparseToDense",category:"slice_join",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!1,notSupported:!0}]},{tfOpName:"TensorScatterUpdate",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],G0=Object.freeze(Object.defineProperty({__proto__:null,json:q0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const M0=[{tfOpName:"SparseFillEmptyRows",category:"sparse",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"denseShape",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}]},{tfOpName:"SparseReshape",category:"sparse",inputs:[{start:0,name:"inputIndices",type:"tensor"},{start:1,name:"inputShape",type:"tensor"},{start:2,name:"newShape",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SparseSegmentMean",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]},{tfOpName:"SparseSegmentSum",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]}],K0=Object.freeze(Object.defineProperty({__proto__:null,json:M0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const H0=[{tfOpName:"FFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"RFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]},{tfOpName:"IRFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]}],X0=Object.freeze(Object.defineProperty({__proto__:null,json:H0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const J0=[{tfOpName:"StaticRegexReplace",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"pattern",name:"pattern",type:"string"},{tfName:"rewrite",name:"rewrite",type:"string"},{tfName:"replace_global",name:"replaceGlobal",type:"bool"}]},{tfOpName:"StringNGrams",category:"string",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"dataSplits",type:"tensor"}],attrs:[{tfName:"separator",name:"separator",type:"string"},{tfName:"ngram_widths",name:"nGramWidths",type:"number[]"},{tfName:"left_pad",name:"leftPad",type:"string"},{tfName:"right_pad",name:"rightPad",type:"string"},{tfName:"pad_width",name:"padWidth",type:"number"},{tfName:"preserve_short_sequences",name:"preserveShortSequences",type:"bool"}],outputs:["ngrams","ngrams_splits"]},{tfOpName:"StringSplit",category:"string",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"delimiter",type:"tensor"}],attrs:[{tfName:"skip_empty",name:"skipEmpty",type:"bool"}],outputs:["indices","values","shape"]},{tfOpName:"StringToHashBucketFast",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"num_buckets",name:"numBuckets",type:"number"}]}],Y0=Object.freeze(Object.defineProperty({__proto__:null,json:J0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Z0=[{tfOpName:"Cast",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"SrcT",name:"sdtype",type:"dtype",notSupported:!0},{tfName:"DstT",name:"dtype",type:"dtype"}]},{tfOpName:"ExpandDims",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"MirrorPad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"mode",name:"mode",type:"string"}]},{tfOpName:"Pad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"constant_value",name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"PadV2",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"},{start:2,name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"Reshape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"EnsureShape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"Squeeze",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"axis",tfDeprecatedName:"squeeze_dims",name:"axis",type:"number[]"}]},{tfOpName:"SpaceToBatchND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"paddings",type:"number[]"}]},{tfOpName:"BatchToSpaceND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"crops",type:"number[]"}]},{tfOpName:"DepthToSpace",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"block_size",name:"blockSize",type:"number"},{tfName:"data_format",name:"dataFormat",type:"string"}]},{tfOpName:"BroadcastTo",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}],attrs:[]},{tfOpName:"BroadcastArgs",category:"transformation",inputs:[{start:0,name:"s0",type:"tensor"},{start:1,name:"s1",type:"tensor"}],attrs:[]}],Q0=Object.freeze(Object.defineProperty({__proto__:null,json:Z0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class jr{static get Instance(){return this._instance||(this._instance=new this)}constructor(){const t=[y0,w0,T0,$0,E0,_0,I0,D0,F0,B0,P0,z0,U0,W0,G0,K0,X0,Y0,Q0],n=[].concat(...t.map(s=>s.json));this.opMappers=n.reduce((s,r)=>(s[r.tfOpName]=r,s),{})}transformGraph(t,n={}){const s=t.node,r=[],a=[],o=[],i=s.reduce((T,N)=>(T[N.name]=this.mapNode(N),N.op.startsWith("Placeholder")?r.push(T[N.name]):N.op==="Const"?a.push(T[N.name]):(N.input==null||N.input.length===0)&&o.push(T[N.name]),T),{});let u=[];const c=[];let h={},p={};n!=null&&(h=this.mapSignatureEntries(n.inputs),p=this.mapSignatureEntries(n.outputs));const f=Object.keys(i);f.forEach(T=>{const N=i[T];N.inputNames.forEach((S,I)=>{const[A,,k]=zt(S),_=i[A];if(_.outputs!=null){const x=_.outputs.indexOf(k);if(x!==-1){const D=`${A}:${x}`;N.inputNames[I]=D}}N.inputs.push(_),_.children.push(N)})}),Object.keys(p).length===0?f.forEach(T=>{const N=i[T];N.children.length===0&&c.push(N)}):Object.keys(p).forEach(T=>{const[N]=zt(T),S=i[N];S!=null&&(S.signatureKey=p[T],c.push(S))}),Object.keys(h).length>0?Object.keys(h).forEach(T=>{const[N]=zt(T),S=i[N];S&&(S.signatureKey=h[T],u.push(S))}):u=r;let d={};t.library!=null&&t.library.function!=null&&(d=t.library.function.reduce((T,N)=>(T[N.signature.name]=this.mapFunction(N),T),{}));const y={nodes:i,inputs:u,outputs:c,weights:a,placeholders:r,signature:n,functions:d};return o.length>0&&(y.initNodes=o),y}mapSignatureEntries(t){return Object.keys(t||{}).reduce((n,s)=>(n[t[s].name]=s,n),{})}mapNode(t){const n=Vo(t.op)||this.opMappers[t.op]||{};t.attr==null&&(t.attr={});const s={name:t.name,op:t.op,category:n.category,inputNames:(t.input||[]).map(r=>r.startsWith("^")?r.slice(1):r),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:t.attr,outputs:n.outputs};return n.inputs!=null&&(s.inputParams=n.inputs.reduce((r,a)=>(r[a.name]={type:a.type,inputIndexStart:a.start,inputIndexEnd:a.end},r),{})),n.attrs!=null&&(s.attrParams=n.attrs.reduce((r,a)=>{const o=a.type;let i;switch(a.type){case"string":i=$s(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=$s(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"string[]":i=As(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=As(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"number":i=Es(t.attr,a.tfName,a.defaultValue||0),i===void 0&&a.tfDeprecatedName&&(i=Es(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"number[]":i=Is(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Is(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"bool":i=ks(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=ks(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"bool[]":i=Os(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Os(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"shape":i=xs(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=xs(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"shape[]":i=Ds(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Ds(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"dtype":i=vs(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=vs(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"dtype[]":i=_s(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=_s(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"func":i=Wr(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Wr(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"tensor":case"tensors":break;default:throw new Error(`Unsupported param type: ${a.type} for op: ${t.op}`)}return r[a.name]={value:i,type:o},r},{})),s}mapFunction(t){const n=t.nodeDef,s=[],r=[];let a={};n!=null&&(a=n.reduce((p,f)=>(p[f.name]=this.mapNode(f),f.op==="Const"&&r.push(p[f.name]),p),{}));const o=[],i=[];t.signature.inputArg.forEach(p=>{const[f]=zt(p.name),d={name:f,op:"Placeholder",inputs:[],inputNames:[],category:"graph",inputParams:{},attrParams:{dtype:{value:dr(p.type),type:"dtype"}},children:[]};d.signatureKey=p.name,o.push(d),a[f]=d}),Object.keys(a).forEach(p=>{const f=a[p];f.inputNames.forEach((d,y)=>{const[T,,N]=zt(d),S=a[T];if(S.outputs!=null){const I=S.outputs.indexOf(N);if(I!==-1){const A=`${T}:${I}`;f.inputNames[y]=A}}f.inputs.push(S),S.children.push(f)})});const c=t.ret;t.signature.outputArg.forEach(p=>{const[f,d]=zt(c[p.name]),y=a[f];y!=null&&(y.defaultOutput=d,i.push(y))});const h=this.mapArgsToSignature(t);return{nodes:a,inputs:o,outputs:i,weights:r,placeholders:s,signature:h}}mapArgsToSignature(t){return{methodName:t.signature.name,inputs:t.signature.inputArg.reduce((n,s)=>(n[s.name]=this.mapArgToTensorInfo(s),n),{}),outputs:t.signature.outputArg.reduce((n,s)=>(n[s.name]=this.mapArgToTensorInfo(s,t.ret),n),{})}}mapArgToTensorInfo(t,n){let s=t.name;return n!=null&&(s=n[s]),{name:s,dtype:t.type}}}function t1(e){const t=C().global;if(typeof t.atob<"u")return t.atob(e);if(typeof Buffer<"u")return new Buffer(e,"base64").toString();throw new Error("Unable to decode base64 in this environment. Missing built-in atob() or Buffer()")}function Uo(e,t){const n=Array.isArray(e)?String.fromCharCode.apply(null,e):t1(e);return t?n:n.toLowerCase()}function $s(e,t,n,s=!1){const r=e[t];return r!=null?Uo(r.s,s):n}function ks(e,t,n){const s=e[t];return s?s.b:n}function Es(e,t,n){const s=e[t]||{},r=s.i!=null?s.i:s.f!=null?s.f:n;return typeof r=="number"?r:parseInt(r,10)}function dr(e){switch(typeof e=="string"&&(e=dt[e]),e){case dt.DT_FLOAT:case dt.DT_HALF:return"float32";case dt.DT_INT32:case dt.DT_INT64:case dt.DT_INT8:case dt.DT_UINT8:return"int32";case dt.DT_BOOL:return"bool";case dt.DT_DOUBLE:return"float32";case dt.DT_STRING:return"string";case dt.DT_COMPLEX64:case dt.DT_COMPLEX128:return"complex64";default:return null}}function Wr(e,t,n){const s=e[t];return s&&s.func?s.func.name:n}function vs(e,t,n){const s=e[t];return s&&s.type?dr(s.type):n}function _s(e,t,n){const s=e[t];return s&&s.list&&s.list.type?s.list.type.map(r=>dr(r)):n}function jo(e){if(!e.unknownRank)return e.dim!=null?e.dim.map(t=>typeof t.size=="number"?t.size:parseInt(t.size,10)):[]}function xs(e,t,n){const s=e[t];return s&&s.shape?jo(s.shape):n}function Is(e,t,n){const s=e[t];return s?((s.list.f&&s.list.f.length?s.list.f:s.list.i)||[]).map(r=>typeof r=="number"?r:parseInt(r,10)):n}function As(e,t,n,s=!1){const r=e[t];return r&&r.list&&r.list.s?r.list.s.map(a=>Uo(a,s)):n}function Ds(e,t,n){const s=e[t];return s&&s.list&&s.list.shape?s.list.shape.map(r=>jo(r)):n}function Os(e,t,n){const s=e[t];return s&&s.list&&s.list.b?s.list.b:n}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class e1{constructor(t,n,s){this.node=t,this.tensorMap=n,this.context=s,this.inputs=[],this.attrs={},this.inputs=t.inputNames.map(r=>this.getInput(r)),t.rawAttrs!=null&&(this.attrs=Object.keys(t.rawAttrs).reduce((r,a)=>(r[a]=this.getAttr(a),r),{}))}getInput(t){return at(t,this.tensorMap,this.context)}getAttr(t,n){const s=this.node.rawAttrs[t];if(s.tensor!=null)return at(t,this.tensorMap,this.context);if(s.i!=null||s.f!=null)return Es(this.node.rawAttrs,t,n);if(s.s!=null)return $s(this.node.rawAttrs,t,n);if(s.b!=null)return ks(this.node.rawAttrs,t,n);if(s.shape!=null)return xs(this.node.rawAttrs,t,n);if(s.type!=null)return vs(this.node.rawAttrs,t,n);if(s.list!=null){if(s.list.i!=null||s.list.f!=null)return Is(this.node.rawAttrs,t,n);if(s.list.s!=null)return As(this.node.rawAttrs,t,n);if(s.list.shape!=null)return Ds(this.node.rawAttrs,t,n);if(s.list.b!=null)return Os(this.node.rawAttrs,t,n);if(s.list.type!=null)return _s(this.node.rawAttrs,t,n)}return n}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ot=Object.freeze(Object.defineProperty({__proto__:null,OP_SCOPE_SUFFIX:Da,abs:Nt,acos:Ah,acosh:Oh,add:F,addN:Ch,all:Rh,any:Lh,argMax:Vh,argMin:jh,asin:qh,asinh:Mh,atan:Hh,atan2:Jh,atanh:Zh,avgPool:Ma,avgPool3d:cp,basicLSTMCell:gp,batchNorm:jn,batchNorm2d:Tp,batchNorm3d:$p,batchNorm4d:Ep,batchToSpaceND:Ka,bincount:Ha,bitwiseAnd:xp,booleanMaskAsync:by,broadcastArgs:Ap,broadcastTo:Sn,buffer:jt,cast:tt,ceil:Fp,clipByValue:Bp,clone:Yt,complex:ee,concat:ct,concat1d:Pp,concat2d:zp,concat3d:Up,concat4d:Wp,conv1d:Mp,conv2d:Wn,conv2dTranspose:Xp,conv3d:Yp,conv3dTranspose:ef,cos:sf,cosh:af,cosineWindow:or,cumprod:uf,cumsum:lf,denseBincount:pf,depthToSpace:mf,depthwiseConv2d:qs,diag:yf,dilation2d:wf,div:K,divNoNan:Ef,dot:_f,dropout:Dy,einsum:Oe,elu:Za,enclosingPowerOfTwo:_o,ensureShape:Df,equal:Ya,erf:Ff,euclideanNorm:Wf,exp:be,expandDims:Kt,expm1:Kf,eye:eo,fft:nr,fill:dn,floor:no,floorDiv:qa,fused:My,gather:so,gatherND:xy,greater:Mn,greaterEqual:ro,ifft:Cn,imag:Kn,image:xw,inTopKAsync:Fy,irfft:To,isFinite:nm,isInf:rm,isNaN:om,leakyRelu:ao,less:ws,lessEqual:Gs,linalg:Iw,linspace:lm,localResponseNormalization:pm,log:an,log1p:oo,logSigmoid:wm,logSoftmax:Sm,logSumExp:uo,logicalAnd:Dn,logicalNot:co,logicalOr:lo,logicalXor:xm,losses:Aw,lowerBound:Am,matMul:U,max:Pe,maxPool:ho,maxPool3d:Fm,maxPoolWithArgmax:Bm,maximum:po,mean:On,meshgrid:Lm,min:bs,minimum:Fn,mirrorPad:Um,mod:Wm,moments:Gm,movingAverage:Ty,mul:v,multiRNNCell:Km,multinomial:Xm,neg:Rt,norm:Gn,notEqual:fo,oneHot:Zm,ones:fe,onesLike:td,op:b,outerProduct:nd,pad:gn,pad1d:ad,pad2d:id,pad3d:cd,pad4d:hd,pool:gd,pow:rn,prelu:go,print:Wa,prod:wd,raggedGather:Td,raggedRange:$d,raggedTensorToTensor:Ed,rand:_d,randomGamma:Kd,randomNormal:bo,randomStandardNormal:Jd,randomUniform:er,randomUniformInt:Qd,range:on,real:un,reciprocal:ng,relu:Xn,relu6:wo,reshape:$,reverse:we,reverse1d:ig,reverse2d:cg,reverse3d:hg,reverse4d:fg,rfft:sr,round:No,rsqrt:gg,scalar:z,scatterND:$y,searchSorted:Ms,selu:bg,separableConv2d:Ng,setdiff1dAsync:Sg,sigmoid:Re,sign:kg,signal:_w,sin:vg,sinh:xg,slice:q,slice1d:Ag,slice2d:Og,slice3d:Cg,slice4d:Rg,softmax:Lg,softplus:io,spaceToBatchND:mo,sparse:Dw,sparseToDense:vy,spectral:vw,split:cn,sqrt:Wt,square:At,squaredDifference:So,squeeze:rr,stack:Gt,step:$o,stridedSlice:Xg,string:Ow,sub:P,sum:M,tan:Yg,tanh:ys,tensor:Bt,tensor1d:xt,tensor2d:Ze,tensor3d:ko,tensor4d:Zg,tensor5d:Qg,tensor6d:ty,tensorScatterUpdate:sy,tile:Ye,topk:ay,transpose:Ns,truncatedNormal:iy,unique:cy,unsortedSegmentSum:hy,unstack:ve,upperBound:fy,variable:my,where:Zt,whereAsync:vo,zeros:qe,zerosLike:Tt},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const n1=(e,t,n,s=ot)=>{switch(e.op){case"BiasAdd":case"AddV2":case"Add":return[s.add(l("a",e,t,n),l("b",e,t,n))];case"AddN":return[s.addN(l("tensors",e,t,n))];case"FloorMod":case"Mod":return[s.mod(l("a",e,t,n),l("b",e,t,n))];case"Mul":return[s.mul(l("a",e,t,n),l("b",e,t,n))];case"RealDiv":case"Div":return[s.div(l("a",e,t,n),l("b",e,t,n))];case"DivNoNan":return[s.divNoNan(l("a",e,t,n),l("b",e,t,n))];case"FloorDiv":return[s.floorDiv(l("a",e,t,n),l("b",e,t,n))];case"Sub":return[s.sub(l("a",e,t,n),l("b",e,t,n))];case"Minimum":return[s.minimum(l("a",e,t,n),l("b",e,t,n))];case"Maximum":return[s.maximum(l("a",e,t,n),l("b",e,t,n))];case"Pow":return[s.pow(l("a",e,t,n),l("b",e,t,n))];case"SquaredDifference":return[s.squaredDifference(l("a",e,t,n),l("b",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const s1=(e,t,n,s=ot)=>{switch(e.op){case"Abs":case"ComplexAbs":return[s.abs(l("x",e,t,n))];case"Acos":return[s.acos(l("x",e,t,n))];case"Acosh":return[s.acosh(l("x",e,t,n))];case"Asin":return[s.asin(l("x",e,t,n))];case"Asinh":return[s.asinh(l("x",e,t,n))];case"Atan":return[s.atan(l("x",e,t,n))];case"Atan2":return[s.atan2(l("x",e,t,n),l("y",e,t,n))];case"Atanh":return[s.atanh(l("x",e,t,n))];case"Ceil":return[s.ceil(l("x",e,t,n))];case"Complex":return[s.complex(l("real",e,t,n),l("imag",e,t,n))];case"Cos":return[s.cos(l("x",e,t,n))];case"Cosh":return[s.cosh(l("x",e,t,n))];case"Elu":return[s.elu(l("x",e,t,n))];case"Erf":return[s.erf(l("x",e,t,n))];case"Exp":return[s.exp(l("x",e,t,n))];case"Expm1":return[s.expm1(l("x",e,t,n))];case"Floor":return[s.floor(l("x",e,t,n))];case"Log":return[s.log(l("x",e,t,n))];case"Log1p":return[s.log1p(l("x",e,t,n))];case"Imag":return[s.imag(l("x",e,t,n))];case"Neg":return[s.neg(l("x",e,t,n))];case"Reciprocal":return[s.reciprocal(l("x",e,t,n))];case"Real":return[s.real(l("x",e,t,n))];case"Relu":return[s.relu(l("x",e,t,n))];case"Round":return[s.round(l("x",e,t,n))];case"Selu":return[s.selu(l("x",e,t,n))];case"Sigmoid":return[s.sigmoid(l("x",e,t,n))];case"Sin":return[s.sin(l("x",e,t,n))];case"Sign":return[s.sign(l("x",e,t,n))];case"Sinh":return[s.sinh(l("x",e,t,n))];case"Softplus":return[s.softplus(l("x",e,t,n))];case"Sqrt":return[s.sqrt(l("x",e,t,n))];case"Square":return[s.square(l("x",e,t,n))];case"Tanh":return[s.tanh(l("x",e,t,n))];case"Tan":return[s.tan(l("x",e,t,n))];case"ClipByValue":return[s.clipByValue(l("x",e,t,n),l("clipValueMin",e,t,n),l("clipValueMax",e,t,n))];case"Relu6":return[s.relu6(l("x",e,t,n))];case"Rsqrt":return[s.rsqrt(at(e.inputNames[0],t,n))];case"LeakyRelu":return[s.leakyRelu(l("x",e,t,n),l("alpha",e,t,n))];case"Prelu":return[s.prelu(l("x",e,t,n),l("alpha",e,t,n))];case"IsNan":return[s.isNaN(at(e.inputNames[0],t,n))];case"IsInf":return[s.isInf(at(e.inputNames[0],t,n))];case"IsFinite":return[s.isFinite(at(e.inputNames[0],t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kt(e,t,n=""){if(!(typeof e=="number"||typeof t=="number")){g(e.length===t.length,()=>n+` Shapes ${e} and ${t} must match`);for(let s=0;s<e.length;s++){const r=e[s],a=t[s];g(r<0||a<0||r===a,()=>n+` Shapes ${e} and ${t} must match`)}}}function qr(e){return!(typeof e=="number"||e.some(t=>t<0))}function Ke(e,t,n){let s=Fs(e,n);const r=!qr(s);if(r&&t.length===0)throw new Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${s}`);if(r&&t.forEach(a=>{s=Fs(a.shape,s)}),!qr(s))throw new Error(`Non-fully-defined elementShape: ${s}`);return s}function Fs(e,t){if(typeof e=="number")return t;if(typeof t=="number")return e;if(e.length!==t.length)throw new Error(`Incompatible ranks during merge: ${e} vs. ${t}`);const n=[];for(let s=0;s<e.length;++s){const r=e[s],a=t[s];if(r>=0&&a>=0&&r!==a)throw new Error(`Incompatible shape during merge: ${e} vs. ${t}`);n[s]=r>=0?r:a}return n}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class r1{constructor(t,n,s,r,a,o,i){this.name=t,this.dtype=n,this.maxSize=s,this.elementShape=r,this.identicalElementShapes=a,this.dynamicSize=o,this.clearAfterRead=i,this.tensors=[],this.closed_=!1,this.idTensor=z(0),Ct(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(t){this.tensors.forEach(n=>{(t==null||!t.has(n.tensor.id))&&n.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(t){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(t<0||t>=this.size())throw new Error(`Tried to read from index ${t}, but array size is: ${this.size()}`);const n=this.tensors[t];if(n.cleared)throw new Error(`TensorArray ${this.name}: Could not read index ${t} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(n.cleared=!0),n.read=!0,n.tensor}readMany(t){return t.map(n=>this.read(n))}write(t,n){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(t<0||!this.dynamicSize&&t>=this.maxSize)throw new Error(`Tried to write to index ${t}, but array is not resizeable and size is: ${this.maxSize}`);const s=this.tensors[t]||{};if(n.dtype!==this.dtype)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${t},
          because the value dtype is ${n.dtype}, but TensorArray dtype is ${this.dtype}.`);if(this.size()===0&&(this.elementShape==null||this.elementShape.length===0)&&(this.elementShape=n.shape),kt(this.elementShape,n.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${t}.`),s.read)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${t}, because it has already been read.`);if(s.written)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${t}, because it has already been written.`);s.tensor=n,Ct(n),s.written=!0,this.tensors[t]=s}writeMany(t,n){if(t.length!==n.length)throw new Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${t.length} is not the same as tensors size: ${n.length}.`);t.forEach((s,r)=>this.write(s,n[r]))}gather(t,n){if(n&&n!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${n}`);if(t)t=t.slice(0,this.size());else{t=[];for(let r=0;r<this.size();r++)t.push(r)}if(t.length===0)return Bt([],[0].concat(this.elementShape));const s=this.readMany(t);return kt(this.elementShape,s[0].shape,"TensorArray shape mismatch: "),Gt(s,0)}concat(t){if(t&&t!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${t}`);if(this.size()===0)return Bt([],[0].concat(this.elementShape));const n=[];for(let r=0;r<this.size();r++)n.push(r);const s=this.readMany(n);return kt(this.elementShape,s[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${s[0].shape})`),ct(s,0)}scatter(t,n){if(n.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${n.dtype}`);if(t.length!==n.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${t.length} vs. ${n.shape[0]}`);const s=Math.max(...t);if(!this.dynamicSize&&s>=this.maxSize)throw new Error(`Max index must be < array size (${s}  vs. ${this.maxSize})`);this.writeMany(t,ve(n,0))}split(t,n){if(n.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${n.dtype}`);let s=0;const r=t.map(u=>(s+=u,s));if(s!==n.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${s}, and tensor's shape is: ${n.shape}`);if(!this.dynamicSize&&t.length!==this.maxSize)throw new Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${t.length}), and the TensorArray is not marked as dynamically resizeable`);const a=s===0?0:n.size/s,o=[];V(()=>{n=$(n,[1,s,a]);for(let u=0;u<t.length;++u){const h=[0,u===0?0:r[u-1],0],p=[1,t[u],a];o[u]=$(q(n,h,p),this.elementShape)}return o});const i=[];for(let u=0;u<t.length;u++)i[u]=u;this.writeMany(i,o)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Te{get id(){return this.idTensor.id}constructor(t,n,s,r=-1){this.tensors=t,this.elementShape=n,this.elementDtype=s,t!=null&&t.forEach(a=>{if(s!==a.dtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${a.dtype}`);kt(n,a.shape,"TensorList shape mismatch: "),Ct(a)}),this.idTensor=z(0),this.maxNumElements=r,Ct(this.idTensor)}copy(){return new Te([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(t){this.tensors.forEach(n=>{(t==null||!t.has(n.id))&&n.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(t,n,s=-1){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);if(s!==-1&&this.tensors.length!==s)throw new Error(`Operation expected a list with ${s} elements but got a list with ${this.tensors.length} elements.`);kt(t,this.elementShape,"TensorList shape mismatch: ");const r=Ke(this.elementShape,this.tensors,t);return V(()=>{const a=this.tensors.map(o=>$(o,r));return Gt(a,0)})}popBack(t,n){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);if(this.size()===0)throw new Error("Trying to pop from an empty list.");const s=Ke(this.elementShape,this.tensors,t),r=this.tensors.pop();return r.kept=!1,kt(r.shape,t,"TensorList shape mismatch: "),$(r,s)}pushBack(t){if(t.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t.dtype}, but list elements ${this.elementDtype}`);if(kt(t.shape,this.elementShape,"TensorList shape mismatch: "),this.maxNumElements===this.size())throw new Error("Trying to push element into a full list.");Ct(t),this.tensors.push(t)}resize(t){if(t<0)throw new Error(`TensorListResize expects size to be non-negative. Got: ${t}`);if(this.maxNumElements!==-1&&t>this.maxNumElements)throw new Error(`TensorListResize input size ${t} is greater maxNumElement ${this.maxNumElements}.`);const n=new Te([],this.elementShape,this.elementDtype,this.maxNumElements);n.tensors.length=t;for(let s=0;s<Math.min(this.tensors.length,t);++s)n.tensors[s]=this.tensors[s];return n}getItem(t,n,s){if(s!==this.elementDtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${this.elementDtype}`);if(t<0||t>this.tensors.length)throw new Error(`Trying to access element ${t} in a list with ${this.tensors.length} elements.`);if(this.tensors[t]==null)throw new Error(`element at index ${t} is null.`);kt(this.tensors[t].shape,n,"TensorList shape mismatch: ");const r=Ke(this.elementShape,this.tensors,n);return $(this.tensors[t],r)}setItem(t,n){if(n.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n.dtype}, but list elements ${this.elementDtype}`);if(t<0||this.maxNumElements!==-1&&t>=this.maxNumElements)throw new Error(`Trying to set element ${t} in a list with max ${this.maxNumElements} elements.`);kt(this.elementShape,n.shape,"TensorList shape mismatch: "),Ct(n),this.tensors[t]!=null&&(this.tensors[t].kept=!1),this.tensors[t]=n}gather(t,n,s){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);kt(this.elementShape,s,"TensorList shape mismatch: "),t=t.slice(0,this.size());const r=Ke(this.elementShape,this.tensors,s);return t.length===0?Bt([],[0].concat(r)):V(()=>{const a=t.map(o=>$(this.tensors[o],r));return Gt(a,0)})}concat(t,n){if(t&&t!==this.elementDtype)throw new Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${t}`);kt(this.elementShape,n,"TensorList shape mismatch: ");const s=Ke(this.elementShape,this.tensors,n);return this.size()===0?Bt([],[0].concat(s)):V(()=>{const r=this.tensors.map(a=>$(a,s));return ct(r,0)})}}function a1(e,t,n){const s=e.dtype;if(e.shape.length<1)throw new Error(`Tensor must be at least a vector, but saw shape: ${e.shape}`);if(e.dtype!==n)throw new Error(`Invalid data types; op elements ${e.dtype}, but list elements ${n}`);const r=e.shape.slice(1);kt(r,t,"TensorList shape mismatch: ");const a=ve(e);return new Te(a,t,s)}function o1(e,t,n,s){return new Te([],e,t,s)}function i1(e,t,n,s){if(t.length!==e.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${t.length} vs. ${e.shape[0]}`);const r=Math.max(...t);if(s!=null&&s!==-1&&r>=s)throw new Error(`Max index must be < array size (${r}  vs. ${s})`);const a=new Te([],n,e.dtype,s),o=ve(e,0);return t.forEach((i,u)=>{a.setItem(i,o[u])}),a}function u1(e,t,n){let s=0;const r=t.map(h=>(s+=h,s));if(s!==e.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${s}, and tensor's shape is: ${e.shape}`);const a=e.shape.slice(1),o=Fs(a,n),i=s===0?0:e.size/s,u=V(()=>{const h=[];e=$(e,[1,s,i]);for(let p=0;p<t.length;++p){const d=[0,p===0?0:r[p-1],0],y=[1,t[p],i];h[p]=$(q(e,d,y),o)}return e.dispose(),h}),c=new Te([],n,e.dtype,t.length);for(let h=0;h<u.length;h++)c.setItem(h,u[h]);return c}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const c1=async(e,t,n)=>{switch(e.op){case"If":case"StatelessIf":{const s=l("thenBranch",e,t,n),r=l("elseBranch",e,t,n),a=l("cond",e,t,n),o=l("args",e,t,n);return(await a.data())[0]?n.functionMap[s].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap):n.functionMap[r].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap)}case"While":case"StatelessWhile":{const s=l("body",e,t,n),r=l("cond",e,t,n),a=l("args",e,t,n),o=await n.functionMap[r].executeFunctionAsync(a,n.tensorArrayMap,n.tensorListMap),i=a.map(h=>h.id);let u=await o[0].data();o.forEach(h=>{!h.kept&&i.indexOf(h.id)===-1&&h.dispose()});let c=a;for(;u[0];){const h=c;c=await n.functionMap[s].executeFunctionAsync(c,n.tensorArrayMap,n.tensorListMap);const p=c.map(d=>d.id);h.forEach(d=>{!d.kept&&i.indexOf(d.id)===-1&&p.indexOf(d.id)===-1&&d.dispose()});const f=await n.functionMap[r].executeFunctionAsync(c,n.tensorArrayMap,n.tensorListMap);u=await f[0].data(),f.forEach(d=>{!d.kept&&i.indexOf(d.id)===-1&&p.indexOf(d.id)===-1&&d.dispose()})}return c}case"LoopCond":{const s=l("pred",e,t,n);return[Vt(s)]}case"Switch":{const s=l("pred",e,t,n);let r=l("data",e,t,n);return r.kept||(r=Vt(r)),(await s.data())[0]?[void 0,r]:[r,void 0]}case"Merge":{const s=e.inputNames.find(r=>at(r,t,n)!==void 0);if(s){const r=at(s,t,n);return[Vt(r)]}return}case"Enter":{const s=l("frameName",e,t,n),r=l("tensor",e,t,n);return n.enterFrame(s),[Vt(r)]}case"Exit":{const s=l("tensor",e,t,n);return n.exitFrame(),[Vt(s)]}case"NextIteration":{const s=l("tensor",e,t,n);return n.nextIteration(),[Vt(s)]}case"TensorArrayV3":{const s=l("size",e,t,n),r=l("dtype",e,t,n),a=l("elementShape",e,t,n),o=l("dynamicSize",e,t,n),i=l("clearAfterRead",e,t,n),u=l("identicalElementShapes",e,t,n),c=l("name",e,t,n),h=new r1(c,r,s,a,u,o,i);return n.addTensorArray(h),[h.idTensor,z(1)]}case"TensorArrayWriteV3":{const s=l("tensorArrayId",e,t,n),r=l("index",e,t,n),a=l("tensor",e,t,n),o=n.getTensorArray(s.id);return o.write(r,a),[o.idTensor]}case"TensorArrayReadV3":{const s=l("tensorArrayId",e,t,n),r=l("index",e,t,n);return[n.getTensorArray(s.id).read(r)]}case"TensorArrayGatherV3":{const s=l("tensorArrayId",e,t,n),r=l("indices",e,t,n),a=l("dtype",e,t,n);return[n.getTensorArray(s.id).gather(r,a)]}case"TensorArrayScatterV3":{const s=l("tensorArrayId",e,t,n),r=l("indices",e,t,n),a=l("tensor",e,t,n),o=n.getTensorArray(s.id);return o.scatter(r,a),[o.idTensor]}case"TensorArrayConcatV3":{const s=l("tensorArrayId",e,t,n),r=n.getTensorArray(s.id),a=l("dtype",e,t,n);return[r.concat(a)]}case"TensorArraySplitV3":{const s=l("tensorArrayId",e,t,n),r=l("tensor",e,t,n),a=l("lengths",e,t,n),o=n.getTensorArray(s.id);return o.split(a,r),[o.idTensor]}case"TensorArraySizeV3":{const s=l("tensorArrayId",e,t,n),r=n.getTensorArray(s.id);return[z(r.size(),"int32")]}case"TensorArrayCloseV3":{const s=l("tensorArrayId",e,t,n),r=n.getTensorArray(s.id);return r.clearAndClose(),[r.idTensor]}case"TensorListSetItem":{const s=l("tensorListId",e,t,n),r=l("index",e,t,n),a=l("tensor",e,t,n),o=n.getTensorList(s.id);return o.setItem(r,a),[o.idTensor]}case"TensorListGetItem":{const s=l("tensorListId",e,t,n),r=l("index",e,t,n),a=l("elementShape",e,t,n),o=l("elementDType",e,t,n);return[n.getTensorList(s.id).getItem(r,a,o)]}case"TensorListScatterV2":case"TensorListScatter":{const s=l("indices",e,t,n),r=l("tensor",e,t,n),a=l("elementShape",e,t,n),o=l("numElements",e,t,n),i=i1(r,s,a,o);return n.addTensorList(i),[i.idTensor]}case"TensorListReserve":case"EmptyTensorList":{const s=l("elementShape",e,t,n),r=l("elementDType",e,t,n);let a;e.op==="TensorListReserve"?a="numElements":a="maxNumElements";const o=l(a,e,t,n),i=e.op==="TensorListReserve"?-1:o,u=o1(s,r,o,i);return n.addTensorList(u),[u.idTensor]}case"TensorListGather":{const s=l("tensorListId",e,t,n),r=l("indices",e,t,n),a=l("elementShape",e,t,n),o=l("elementDType",e,t,n);return[n.getTensorList(s.id).gather(r,o,a)]}case"TensorListStack":{const s=l("tensorListId",e,t,n),r=l("elementShape",e,t,n),a=l("elementDType",e,t,n),o=l("numElements",e,t,n);return[n.getTensorList(s.id).stack(r,a,o)]}case"TensorListFromTensor":{const s=l("tensor",e,t,n),r=l("elementShape",e,t,n),a=l("elementDType",e,t,n),o=a1(s,r,a);return n.addTensorList(o),[o.idTensor]}case"TensorListConcat":case"TensorListConcatV2":{const s=l("tensorListId",e,t,n),r=n.getTensorList(s.id),a=l("dtype",e,t,n),o=l("elementShape",e,t,n);return[r.concat(a,o)]}case"TensorListPushBack":{const s=l("tensorListId",e,t,n),r=l("tensor",e,t,n),a=n.getTensorList(s.id);return a.pushBack(r),[a.idTensor]}case"TensorListPopBack":{const s=l("tensorListId",e,t,n),r=l("elementShape",e,t,n),a=l("elementDType",e,t,n);return[n.getTensorList(s.id).popBack(r,a)]}case"TensorListSplit":{const s=l("tensor",e,t,n),r=l("elementShape",e,t,n),a=l("lengths",e,t,n),o=u1(s,a,r);return n.addTensorList(o),[o.idTensor]}case"TensorListLength":{const s=l("tensorListId",e,t,n),r=n.getTensorList(s.id);return[z(r.size(),"int32")]}case"TensorListResize":{const s=l("tensorListId",e,t,n),r=l("size",e,t,n),o=n.getTensorList(s.id).resize(r);return n.addTensorList(o),[o.idTensor]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gr(e,t,n){const[s,r]=l("fusedOps",e,t,n),a=s==="biasadd",o=!a,i=r==="prelu",u=s==="fusedbatchnorm",c=l("numArgs",e,t,n);if(a){if(i&&c!==2)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!i&&a&&c!==1)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.")}if(u)throw new Error("FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported");const h=l("strides",e,t,n),p=$n(e,t,n),f=l("dataFormat",e,t,n).toUpperCase(),d=l("dilations",e,t,n);let[y,T]=l("args",e,t,n);o&&(T=y,y=void 0);const N=l("leakyreluAlpha",e,t,n);return{stride:h,pad:p,dataFormat:f,dilations:d,biasArg:y,preluArg:T,activationFunc:r,leakyreluAlpha:N}}const l1=(e,t,n,s=ot)=>{switch(e.op){case"Conv1D":{const r=l("stride",e,t,n),a=l("pad",e,t,n),o=l("dataFormat",e,t,n).toUpperCase(),i=l("dilation",e,t,n);return[s.conv1d(l("x",e,t,n),l("filter",e,t,n),r,a,o,i)]}case"Conv2D":{const r=l("strides",e,t,n),a=$n(e,t,n),o=l("dataFormat",e,t,n).toUpperCase(),i=l("dilations",e,t,n);return[s.conv2d(l("x",e,t,n),l("filter",e,t,n),[r[1],r[2]],a,o,[i[1],i[2]])]}case"_FusedConv2D":{const{stride:r,pad:a,dataFormat:o,dilations:i,biasArg:u,preluArg:c,activationFunc:h,leakyreluAlpha:p}=Gr(e,t,n);return[s.fused.conv2d({x:l("x",e,t,n),filter:l("filter",e,t,n),strides:[r[1],r[2]],pad:a,dataFormat:o,dilations:[i[1],i[2]],bias:u,activation:h,preluActivationWeights:c,leakyreluAlpha:p})]}case"FusedDepthwiseConv2dNative":{const{stride:r,pad:a,dataFormat:o,dilations:i,biasArg:u,preluArg:c,activationFunc:h,leakyreluAlpha:p}=Gr(e,t,n);return[s.fused.depthwiseConv2d({x:l("x",e,t,n),filter:l("filter",e,t,n),strides:[r[1],r[2]],pad:a,dataFormat:o,dilations:[i[1],i[2]],bias:u,activation:h,preluActivationWeights:c,leakyreluAlpha:p})]}case"Conv2DBackpropInput":case"Conv2dTranspose":{const r=l("outputShape",e,t,n),a=l("strides",e,t,n),o=$n(e,t,n);return[s.conv2dTranspose(l("x",e,t,n),l("filter",e,t,n),r,[a[1],a[2]],o)]}case"DepthwiseConv2dNative":case"DepthwiseConv2d":{const r=l("strides",e,t,n),a=$n(e,t,n),o=l("dilations",e,t,n),i=l("dataFormat",e,t,n).toUpperCase();return[s.depthwiseConv2d(l("input",e,t,n),l("filter",e,t,n),[r[1],r[2]],a,i,[o[1],o[2]])]}case"Conv3D":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("dataFormat",e,t,n).toUpperCase(),i=l("dilations",e,t,n);return[s.conv3d(l("x",e,t,n),l("filter",e,t,n),[r[1],r[2],r[3]],a,o,[i[1],i[2],i[3]])]}case"AvgPool":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("kernelSize",e,t,n);return[s.avgPool(l("x",e,t,n),[o[1],o[2]],[r[1],r[2]],a)]}case"MaxPool":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("kernelSize",e,t,n);return[s.maxPool(l("x",e,t,n),[o[1],o[2]],[r[1],r[2]],a)]}case"MaxPoolWithArgmax":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("kernelSize",e,t,n),i=l("includeBatchInIndex",e,t,n),{result:u,indexes:c}=s.maxPoolWithArgmax(l("x",e,t,n),[o[1],o[2]],[r[1],r[2]],a,i);return[u,c]}case"AvgPool3D":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("kernelSize",e,t,n);return[s.avgPool3d(l("x",e,t,n),[o[1],o[2],o[3]],[r[1],r[2],r[3]],a)]}case"MaxPool3D":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("kernelSize",e,t,n);return[s.maxPool3d(l("x",e,t,n),[o[1],o[2],o[3]],[r[1],r[2],r[3]],a)]}case"Dilation2D":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("dilations",e,t,n),i=r[1],u=r[2],c=o[1],h=o[2];return[s.dilation2d(l("x",e,t,n),l("filter",e,t,n),[i,u],a,[c,h],"NHWC")]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const h1=(e,t,n,s=ot)=>{switch(e.op){case"Fill":{const r=l("shape",e,t,n),a=l("dtype",e,t,n),o=l("value",e,t,n);return[s.fill(r,o,a)]}case"LinSpace":{const r=l("start",e,t,n),a=l("stop",e,t,n),o=l("num",e,t,n);return[s.linspace(r,a,o)]}case"Multinomial":{const r=l("logits",e,t,n),a=l("numSamples",e,t,n),o=l("seed",e,t,n);return[s.multinomial(r,a,o)]}case"OneHot":{const r=l("indices",e,t,n),a=l("depth",e,t,n),o=l("onValue",e,t,n),i=l("offValue",e,t,n),u=l("dtype",e,t,n);return[s.oneHot(r,a,o,i,u)]}case"Ones":return[s.ones(l("shape",e,t,n),l("dtype",e,t,n))];case"OnesLike":return[s.onesLike(l("x",e,t,n))];case"RandomStandardNormal":return[s.randomStandardNormal(l("shape",e,t,n),l("dtype",e,t,n),l("seed",e,t,n))];case"RandomUniform":return[s.randomUniform(l("shape",e,t,n),l("minval",e,t,n),l("maxval",e,t,n),l("dtype",e,t,n))];case"RandomUniformInt":return[s.randomUniformInt(l("shape",e,t,n),l("minval",e,t,n),l("maxval",e,t,n),l("seed",e,t,n))];case"Range":{const r=l("start",e,t,n),a=l("stop",e,t,n),o=l("step",e,t,n);return[s.range(r,a,o,l("dtype",e,t,n))]}case"TruncatedNormal":{const r=l("shape",e,t,n),a=l("mean",e,t,n),o=l("stdDev",e,t,n),i=l("seed",e,t,n);return[s.truncatedNormal(r,a,o,l("dtype",e,t,n),i)]}case"Zeros":return[s.zeros(l("shape",e,t,n),l("dtype",e,t,n))];case"ZerosLike":return[s.zerosLike(l("x",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ns(e,t,n){const s=l("boxes",e,t,n),r=l("scores",e,t,n),a=l("maxOutputSize",e,t,n),o=l("iouThreshold",e,t,n),i=l("scoreThreshold",e,t,n),u=l("softNmsSigma",e,t,n);return{boxes:s,scores:r,maxOutputSize:a,iouThreshold:o,scoreThreshold:i,softNmsSigma:u}}const p1=async(e,t,n,s,r=ot)=>{switch(e.op){case"NonMaxSuppressionV5":{const{boxes:a,scores:o,maxOutputSize:i,iouThreshold:u,scoreThreshold:c,softNmsSigma:h}=ns(e,t,n),p=await r.image.nonMaxSuppressionWithScoreAsync(a,o,i,u,c,h);return[p.selectedIndices,p.selectedScores]}case"NonMaxSuppressionV4":{const{boxes:a,scores:o,maxOutputSize:i,iouThreshold:u,scoreThreshold:c}=ns(e,t,n),h=l("padToMaxOutputSize",e,t,n),p=await r.image.nonMaxSuppressionPaddedAsync(a,o,i,u,c,h);return[p.selectedIndices,p.validOutputs]}case"NonMaxSuppressionV3":case"NonMaxSuppressionV2":{const{boxes:a,scores:o,maxOutputSize:i,iouThreshold:u,scoreThreshold:c}=ns(e,t,n);return[await r.image.nonMaxSuppressionAsync(a,o,i,u,c)]}case"Where":{const a=r.cast(l("condition",e,t,n),"bool"),o=[await r.whereAsync(a)];return a.dispose(),o}case"ListDiff":return r.setdiff1dAsync(l("x",e,t,n),l("y",e,t,n));default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const f1=(e,t,n,s=ot)=>{switch(e.op){case"LowerBound":{const r=l("sortedSequence",e,t,n),a=l("values",e,t,n);return[s.lowerBound(r,a)]}case"TopKV2":{const r=l("x",e,t,n),a=l("k",e,t,n),o=l("sorted",e,t,n),i=s.topk(r,a,o);return[i.values,i.indices]}case"UpperBound":{const r=l("sortedSequence",e,t,n),a=l("values",e,t,n);return[s.upperBound(r,a)]}case"Unique":{const r=l("x",e,t,n),a=s.unique(r);return[a.values,a.indices]}case"UniqueV2":{const r=l("x",e,t,n),a=l("axis",e,t,n),o=s.unique(r,a);return[o.values,o.indices]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const m1=(e,t,n,s=ot)=>{switch(e.op){case"Const":return t[e.name];case"PlaceholderWithDefault":const r=l("default",e,t,n);return[at(e.name,t,n)||r];case"Placeholder":return[at(e.name,t,n)];case"Identity":case"StopGradient":case"FakeQuantWithMinMaxVars":{const h=l("x",e,t,n);return[Vt(h)]}case"IdentityN":return l("x",e,t,n).map(h=>Vt(h));case"Snapshot":const a=l("x",e,t,n);return[Vt(a)];case"Shape":return[s.tensor1d(l("x",e,t,n).shape,"int32")];case"ShapeN":return l("x",e,t,n).map(h=>s.tensor1d(h.shape));case"Size":return[s.scalar(l("x",e,t,n).size,"int32")];case"Rank":return[s.scalar(l("x",e,t,n).rank,"int32")];case"NoOp":return[s.scalar(1)];case"Print":const o=l("x",e,t,n),i=l("data",e,t,n),u=l("message",e,t,n),c=l("summarize",e,t,n);console.warn("The graph has a tf.print() operation,usually used for debugging, which slows down performance."),console.log(u);for(let h=0;h<i.length;h++)console.log(Array.prototype.slice.call(i[h].dataSync()).slice(0,c));return[o];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class d1{get id(){return this.handle.id}constructor(t,n){this.keyDType=t,this.valueDType=n,this.handle=z(0),this.tensorMap=new Map,Ct(this.handle)}clearAndClose(){this.tensorMap.forEach(t=>t.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return z(this.size(),"int32")}async import(t,n){this.checkKeyAndValueTensor(t,n);const s=await t.data();return this.tensorMap.forEach(r=>r.dispose()),this.tensorMap.clear(),V(()=>{const r=ve(n),a=s.length,o=r.length;g(a===o,()=>`The number of elements doesn't match, keys has ${a} elements, the values has ${o} elements.`);for(let i=0;i<a;i++){const u=s[i],c=r[i];Ct(c),this.tensorMap.set(u,c)}return this.handle})}async find(t,n){this.checkKeyAndValueTensor(t,n);const s=await t.data();return V(()=>{const r=[];for(let a=0;a<s.length;a++){const o=s[a],i=this.findWithDefault(o,n);r.push(i)}return Gt(r)})}findWithDefault(t,n){const s=this.tensorMap.get(t);return s??n}checkKeyAndValueTensor(t,n){if(t.dtype!==this.keyDType)throw new Error(`Expect key dtype ${this.keyDType}, but got ${t.dtype}`);if(n.dtype!==this.valueDType)throw new Error(`Expect value dtype ${this.valueDType}, but got ${n.dtype}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const g1=async(e,t,n,s)=>{switch(e.op){case"HashTable":case"HashTableV2":{const r=s.getHashTableHandleByName(e.name);if(r!=null)return[r];{const a=l("keyDType",e,t,n),o=l("valueDType",e,t,n),i=new d1(a,o);return s.addHashTable(e.name,i),[i.handle]}}case"InitializeTable":case"InitializeTableV2":case"LookupTableImport":case"LookupTableImportV2":{const r=l("tableHandle",e,t,n,s),a=l("keys",e,t,n),o=l("values",e,t,n);return[await s.getHashTableById(r.id).import(a,o)]}case"LookupTableFind":case"LookupTableFindV2":{const r=l("tableHandle",e,t,n,s),a=l("keys",e,t,n),o=l("defaultValue",e,t,n);return[await s.getHashTableById(r.id).find(a,o)]}case"LookupTableSize":case"LookupTableSizeV2":{const r=l("tableHandle",e,t,n,s);return[s.getHashTableById(r.id).tensorSize()]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const y1=(e,t,n,s=ot)=>{switch(e.op){case"ResizeBilinear":{const r=l("images",e,t,n),a=l("size",e,t,n),o=l("alignCorners",e,t,n),i=l("halfPixelCenters",e,t,n);return[s.image.resizeBilinear(r,[a[0],a[1]],o,i)]}case"ResizeNearestNeighbor":{const r=l("images",e,t,n),a=l("size",e,t,n),o=l("alignCorners",e,t,n),i=l("halfPixelCenters",e,t,n);return[s.image.resizeNearestNeighbor(r,[a[0],a[1]],o,i)]}case"CropAndResize":{const r=l("image",e,t,n),a=l("boxes",e,t,n),o=l("boxInd",e,t,n),i=l("cropSize",e,t,n),u=l("method",e,t,n),c=l("extrapolationValue",e,t,n);return[s.image.cropAndResize(r,a,o,i,u,c)]}case"ImageProjectiveTransformV3":{const r=l("images",e,t,n),a=l("transforms",e,t,n),o=l("outputShape",e,t,n),i=l("fillValue",e,t,n),u=l("interpolation",e,t,n),c=l("fillMode",e,t,n);return[s.image.transform(r,a,u.toLowerCase(),c.toLowerCase(),i,o)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const b1=(e,t,n,s=ot)=>{switch(e.op){case"Equal":return[s.equal(l("a",e,t,n),l("b",e,t,n))];case"NotEqual":return[s.notEqual(l("a",e,t,n),l("b",e,t,n))];case"Greater":return[s.greater(l("a",e,t,n),l("b",e,t,n))];case"GreaterEqual":return[s.greaterEqual(l("a",e,t,n),l("b",e,t,n))];case"Less":return[s.less(l("a",e,t,n),l("b",e,t,n))];case"LessEqual":return[s.lessEqual(l("a",e,t,n),l("b",e,t,n))];case"LogicalAnd":return[s.logicalAnd(l("a",e,t,n),l("b",e,t,n))];case"LogicalNot":return[s.logicalNot(l("a",e,t,n))];case"LogicalOr":return[s.logicalOr(l("a",e,t,n),l("b",e,t,n))];case"Select":case"SelectV2":return[s.where(l("condition",e,t,n),l("a",e,t,n),l("b",e,t,n))];case"BitwiseAnd":return[s.bitwiseAnd(l("a",e,t,n),l("b",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const w1=(e,t,n,s=ot)=>{switch(e.op){case"BatchMatMul":case"BatchMatMulV2":case"MatMul":return[s.matMul(l("a",e,t,n),l("b",e,t,n),l("transposeA",e,t,n),l("transposeB",e,t,n))];case"Einsum":return[s.einsum(l("equation",e,t,n),...l("tensors",e,t,n))];case"Transpose":return[s.transpose(l("x",e,t,n),l("perm",e,t,n))];case"_FusedMatMul":const[r,a]=l("fusedOps",e,t,n),o=r==="biasadd",i=a==="prelu",u=l("numArgs",e,t,n),c=l("leakyreluAlpha",e,t,n);if(o){if(i&&u!==2)throw new Error("Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!i&&u!==1)throw new Error("Fused MatMul with BiasAdd must have one extra argument: bias.")}const[h,p]=l("args",e,t,n);return[s.fused.matMul({a:l("a",e,t,n),b:l("b",e,t,n),transposeA:l("transposeA",e,t,n),transposeB:l("transposeB",e,t,n),bias:h,activation:a,preluActivationWeights:p,leakyreluAlpha:c})];case"MatrixBandPart":return[s.linalg.bandPart(l("a",e,t,n),l("numLower",e,t,n),l("numUpper",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const N1=(e,t,n,s=ot)=>{switch(e.op){case"EuclideanNorm":return[s.euclideanNorm(l("x",e,t,n),l("axis",e,t,n),l("keepDims",e,t,n))];case"FusedBatchNorm":case"FusedBatchNormV2":return[s.batchNorm(l("x",e,t,n),l("mean",e,t,n),l("variance",e,t,n),l("offset",e,t,n),l("scale",e,t,n),l("epsilon",e,t,n))];case"FusedBatchNormV3":return[s.batchNorm(l("x",e,t,n),l("mean",e,t,n),l("variance",e,t,n),l("offset",e,t,n),l("scale",e,t,n),l("epsilon",e,t,n))];case"LRN":return[s.localResponseNormalization(l("x",e,t,n),l("radius",e,t,n),l("bias",e,t,n),l("alpha",e,t,n),l("beta",e,t,n))];case"Softmax":return[s.softmax(l("x",e,t,n))];case"LogSoftmax":return[s.logSoftmax(l("x",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const T1=(e,t,n,s=ot)=>{switch(e.op){case"RaggedGather":{const{outputNestedSplits:r,outputDenseValues:a}=s.raggedGather(l("paramsNestedSplits",e,t,n),l("paramsDenseValues",e,t,n),l("indices",e,t,n),l("outputRaggedRank",e,t,n));return r.concat(a)}case"RaggedRange":{const{rtNestedSplits:r,rtDenseValues:a}=s.raggedRange(l("starts",e,t,n),l("limits",e,t,n),l("splits",e,t,n));return[r,a]}case"RaggedTensorToTensor":return[s.raggedTensorToTensor(l("shape",e,t,n),l("values",e,t,n),l("defaultValue",e,t,n),l("rowPartitionTensors",e,t,n),l("rowPartitionTypes",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const S1=(e,t,n,s=ot)=>{switch(e.op){case"Max":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.max(l("x",e,t,n),i,u)]}case"Mean":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.mean(l("x",e,t,n),i,u)]}case"Min":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.min(l("x",e,t,n),i,u)]}case"Sum":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.sum(l("x",e,t,n),i,u)]}case"All":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.all(l("x",e,t,n),i,u)]}case"Any":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.any(l("x",e,t,n),i,u)]}case"ArgMax":{const i=l("axis",e,t,n);return[s.argMax(l("x",e,t,n),i)]}case"ArgMin":{const i=l("axis",e,t,n);return[s.argMin(l("x",e,t,n),i)]}case"Prod":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.prod(l("x",e,t,n),i,u)]}case"Cumprod":{const i=l("axis",e,t,n),u=l("exclusive",e,t,n),c=l("reverse",e,t,n);return[s.cumprod(l("x",e,t,n),i,u,c)]}case"Cumsum":{const i=l("axis",e,t,n),u=l("exclusive",e,t,n),c=l("reverse",e,t,n);return[s.cumsum(l("x",e,t,n),i,u,c)]}case"Bincount":const r=l("x",e,t,n),a=l("weights",e,t,n),o=l("size",e,t,n);return[s.bincount(r,a,o)];case"DenseBincount":{const i=l("x",e,t,n),u=l("weights",e,t,n),c=l("size",e,t,n),h=l("binaryOutput",e,t,n);return[s.denseBincount(i,u,c,h)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $1=(e,t,n,s=ot)=>{switch(e.op){case"ConcatV2":case"Concat":{const r=l("n",e,t,n),a=l("axis",e,t,n);let o=l("tensors",e,t,n);return o=o.slice(0,r),[s.concat(o,a)]}case"Gather":{const r=l("x",e,t,n),a=l("indices",e,t,n);return[s.gather(r,s.cast(a,"int32"),0)]}case"GatherV2":{const r=l("axis",e,t,n),a=l("batchDims",e,t,n),o=l("x",e,t,n),i=l("indices",e,t,n);return[s.gather(o,s.cast(i,"int32"),r,a)]}case"Reverse":{const r=l("dims",e,t,n),a=[];for(let i=0;i<r.length;i++)r[i]&&a.push(i);const o=l("x",e,t,n);return[s.reverse(o,a)]}case"ReverseV2":{const r=l("axis",e,t,n),a=l("x",e,t,n);return[s.reverse(a,r)]}case"Slice":{const r=l("begin",e,t,n),a=l("size",e,t,n);return[s.slice(l("x",e,t,n),r,a)]}case"StridedSlice":{const r=l("begin",e,t,n),a=l("end",e,t,n),o=l("strides",e,t,n),i=l("beginMask",e,t,n),u=l("endMask",e,t,n),c=l("ellipsisMask",e,t,n),h=l("newAxisMask",e,t,n),p=l("shrinkAxisMask",e,t,n),f=l("x",e,t,n);return[s.stridedSlice(f,r,a,o,i,u,c,h,p)]}case"Pack":return V(()=>{const r=l("axis",e,t,n),a=l("tensors",e,t,n),o=a[0].shape,i=s.squeeze(a[0]).shape,u=a.map(c=>{const h=Pt(c.shape,o);if(!h&&!Pt(s.squeeze(c).shape,i))throw new Error("the input tensors shape does not match");return h?c:s.reshape(c,o)});return[s.stack(u,r)]});case"Unpack":{const r=l("axis",e,t,n),a=l("tensor",e,t,n);return s.unstack(a,r)}case"Tile":{const r=l("reps",e,t,n);return[s.tile(l("x",e,t,n),r)]}case"Split":case"SplitV":{const r=l("axis",e,t,n),a=l("numOrSizeSplits",e,t,n),o=l("x",e,t,n);return s.split(o,a,r)}case"ScatterNd":{const r=l("indices",e,t,n),a=l("values",e,t,n),o=l("shape",e,t,n);return[s.scatterND(r,a,o)]}case"GatherNd":{const r=l("x",e,t,n),a=l("indices",e,t,n);return[s.gatherND(r,a)]}case"SparseToDense":{const r=l("sparseIndices",e,t,n),a=l("outputShape",e,t,n),o=l("sparseValues",e,t,n),i=l("defaultValue",e,t,n);return[s.sparseToDense(r,o,a,o.dtype===i.dtype?i:s.cast(i,o.dtype))]}case"TensorScatterUpdate":{const r=l("indices",e,t,n),a=l("values",e,t,n),o=l("tensor",e,t,n);return[s.tensorScatterUpdate(o,r,a)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const k1=(e,t,n,s=ot)=>{switch(e.op){case"SparseFillEmptyRows":{const{outputIndices:r,outputValues:a,emptyRowIndicator:o,reverseIndexMap:i}=s.sparse.sparseFillEmptyRows(l("indices",e,t,n),l("values",e,t,n),l("denseShape",e,t,n),l("defaultValue",e,t,n));return[r,a,o,i]}case"SparseReshape":{const{outputIndices:r,outputShape:a}=s.sparse.sparseReshape(l("inputIndices",e,t,n),l("inputShape",e,t,n),l("newShape",e,t,n));return[r,a]}case"SparseSegmentMean":return[s.sparse.sparseSegmentMean(l("data",e,t,n),l("indices",e,t,n),l("segmentIds",e,t,n))];case"SparseSegmentSum":return[s.sparse.sparseSegmentSum(l("data",e,t,n),l("indices",e,t,n),l("segmentIds",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const E1=(e,t,n,s=ot)=>{switch(e.op){case"FFT":return[s.fft(l("x",e,t,n))];case"IFFT":return[s.ifft(l("x",e,t,n))];case"RFFT":return[s.rfft(l("x",e,t,n))];case"IRFFT":return[s.irfft(l("x",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const v1=(e,t,n,s=ot)=>{switch(e.op){case"StaticRegexReplace":return[s.string.staticRegexReplace(l("input",e,t,n),l("pattern",e,t,n),l("rewrite",e,t,n),l("replaceGlobal",e,t,n))];case"StringNGrams":{const{nGrams:r,nGramsSplits:a}=s.string.stringNGrams(l("data",e,t,n),l("dataSplits",e,t,n),l("separator",e,t,n),l("nGramWidths",e,t,n),l("leftPad",e,t,n),l("rightPad",e,t,n),l("padWidth",e,t,n),l("preserveShortSequences",e,t,n));return[r,a]}case"StringSplit":{const{indices:r,values:a,shape:o}=s.string.stringSplit(l("input",e,t,n),l("delimiter",e,t,n),l("skipEmpty",e,t,n));return[r,a,o]}case"StringToHashBucketFast":return[s.string.stringToHashBucketFast(l("input",e,t,n),l("numBuckets",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _1=(e,t,n,s=ot)=>{switch(e.op){case"Cast":return[s.cast(l("x",e,t,n),l("dtype",e,t,n))];case"ExpandDims":{const r=l("axis",e,t,n);return[s.expandDims(l("x",e,t,n),r)]}case"Squeeze":{const r=l("axis",e,t,n);return[s.squeeze(l("x",e,t,n),r)]}case"Reshape":return[s.reshape(l("x",e,t,n),l("shape",e,t,n))];case"EnsureShape":return[s.ensureShape(l("x",e,t,n),l("shape",e,t,n))];case"MirrorPad":return[s.mirrorPad(l("x",e,t,n),l("padding",e,t,n),l("mode",e,t,n))];case"PadV2":case"Pad":return[s.pad(l("x",e,t,n),l("padding",e,t,n),l("constantValue",e,t,n))];case"SpaceToBatchND":{const r=l("blockShape",e,t,n),a=l("paddings",e,t,n);return[s.spaceToBatchND(l("x",e,t,n),r,a)]}case"BatchToSpaceND":{const r=l("blockShape",e,t,n),a=l("crops",e,t,n);return[s.batchToSpaceND(l("x",e,t,n),r,a)]}case"DepthToSpace":{const r=l("blockSize",e,t,n),a=l("dataFormat",e,t,n).toUpperCase();return[s.depthToSpace(l("x",e,t,n),r,a)]}case"BroadcastTo":return[s.broadcastTo(l("x",e,t,n),l("shape",e,t,n))];case"BroadcastArgs":return[s.broadcastArgs(l("s0",e,t,n),l("s1",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mr(e,t,n,s,r=V){const a=((o,i,u)=>{switch(o.category){case"arithmetic":return r(()=>n1(o,i,u));case"basic_math":return r(()=>s1(o,i,u));case"control":return c1(o,i,u);case"convolution":return r(()=>l1(o,i,u));case"creation":return r(()=>h1(o,i,u));case"dynamic":return p1(o,i,u);case"evaluation":return r(()=>f1(o,i,u));case"image":return r(()=>y1(o,i,u));case"graph":return r(()=>m1(o,i,u));case"logical":return r(()=>b1(o,i,u));case"matrices":return r(()=>w1(o,i,u));case"normalization":return r(()=>N1(o,i,u));case"ragged":return r(()=>T1(o,i,u));case"reduction":return r(()=>S1(o,i,u));case"slice_join":return r(()=>$1(o,i,u));case"sparse":return r(()=>k1(o,i,u));case"spectral":return r(()=>E1(o,i,u));case"string":return r(()=>v1(o,i,u));case"transformation":return r(()=>_1(o,i,u));case"hash_table":return g1(o,i,u,s);case"custom":const c=Vo(o.op);if(c&&c.customExecutor)return c.customExecutor(new e1(o,i,u));throw TypeError(`Custom op ${o.op} is not registered.`);default:throw TypeError(`Unknown op '${o.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(e,t,n);return te(a)?a.then(o=>[].concat(o)):[].concat(a)}class Kr{constructor(t={},n={},s={},r={},a){this.weightMap=t,this.tensorArrayMap=n,this.tensorListMap=s,this.functionMap=r,this.parseNodeNameCache=a,this.rootContext={id:0,frameName:"",iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(t,n){return{id:t,frameName:n,iterationId:0}}set currentContext(t){this.contexts!==t&&(this.contexts=t,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){const t=[];for(let n=0;n<this.contexts.length-1;n++){const s=this.contexts.slice(0,this.contexts.length-n);t.push(this.contextIdforContexts(s))}t.push(""),this._currentContextIds=t}contextIdforContexts(t){return t?t.map(n=>n.id===0&&n.iterationId===0?"":`${n.frameName}-${n.iterationId}`).join("/"):""}enterFrame(t){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,t)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(this.contexts&&this.contexts.length>1)this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift();else throw new Error("Cannot exit frame, the context is empty")}nextIteration(){if(this.contexts&&this.contexts.length>0){this.contexts=this.contexts.slice(),this.lastId++;const t=Object.assign({},this.contexts[this.contexts.length-1]);t.iterationId+=1,t.id=this.lastId,this.contexts.splice(-1,1,t),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}else throw new Error("Cannot increase frame iteration, the context is empty")}getWeight(t){return this.weightMap[t]}addTensorArray(t){this.tensorArrayMap[t.id]=t}getTensorArray(t){return this.tensorArrayMap[t]}addTensorList(t){this.tensorListMap[t.id]=t}getTensorList(t){return this.tensorListMap[t]}dispose(t){for(const n in this.tensorArrayMap)this.tensorArrayMap[n].clearAndClose(t);for(const n in this.tensorListMap)this.tensorListMap[n].clearAndClose(t)}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hr(e,t,n,s){const r=new Set,a=[];let o=null,i=null;const u=new Set,c=new Set(Object.keys(e).map(f=>gt(f)[0]));s=s||[];const h=new Set(s.map(f=>gt(f.name)[0])),p=[...t];for(;p.length>0;){const f=p.pop();if((le(f)||B1(f)||R1(f))&&o==null&&(o=f,i=o.children.map(d=>d.name).filter(d=>r.has(d))),r.add(f.name),n[f.name]==null&&!c.has(f.name)&&!h.has(f.name)){if(f.inputs.length===0){a.push(f.name);continue}f.inputs.forEach(d=>{u.has(d.name)||(u.add(d.name),p.push(d))})}}return{inputs:e,outputs:t,usedNodes:r,missingInputs:a,dynamicNode:o,syncInputs:i}}function x1(e,t){const{usedNodes:n,inputs:s}=t,r=Object.keys(s).map(N=>gt(N)[0]).map(N=>e.nodes[N]),a=e.initNodes||[],o=N=>n.has(typeof N=="string"?N:N.name);function i(N){return[...new Map(N.map(S=>[S.name,S])).values()]}const u=i([...r,...e.weights,...a]).filter(o),c=i([...u,...Object.values(e.nodes)]).filter(o),h=new Map(c.map(N=>[N.name,N])),p={};for(const N of c){p[N.name]=p[N.name]||0;for(const S of N.children)o(S)||(p[S.name]=Number.POSITIVE_INFINITY),p[S.name]=(p[S.name]||0)+1}const f=Object.entries(p).filter(([,N])=>N===0).map(([N])=>N),d=[...f];for(;f.length>0;){const N=f.pop(),S=h.get(N);for(const I of S.children.filter(o))--p[I.name]===0&&(d.push(I.name),f.push(I.name))}const y=d.map(N=>h.get(N)),T=I1(y,u);return A1(T,u),T}function I1(e,t){const n=new Map(e.map(o=>[o.name,o])),s=t.map(o=>o.name),r=new Set(s);for(;s.length>0;){const o=s.pop(),i=n.get(o);for(const u of i.children)!n.has(u.name)||r.has(u.name)||(r.add(u.name),s.push(u.name))}return e.filter(o=>r.has(o.name))}class wn extends Error{constructor(t){super(`NodesExecutionOrderError: ${t}`)}}function A1(e,t){const n=new Map(e.map((i,u)=>[i.name,u])),s=new Set(t.map(i=>i.name)),r=i=>s.has(typeof i=="string"?i:i.name),a=new Set(e.map(i=>i.name)),o=i=>a.has(typeof i=="string"?i:i.name);for(const i of e){for(const u of i.children.filter(o)){if(!n.has(u.name))throw new wn(`Child ${u.name} of node ${i.name} is unreachable.`);if(n.get(i.name)>n.get(u.name))throw new wn(`Node ${i.name} is scheduled to run after its child ${u.name}.`)}if(!r(i))for(const u of i.inputs){if(!n.has(u.name))throw new wn(`Input ${u.name} of node ${i.name} is unreachable.`);if(n.get(u.name)>n.get(i.name))throw new wn(`Node ${i.name} is scheduled to run before its input ${u.name}.`)}}}function D1(e){const t=new Map(e.map((i,u)=>[i.name,u])),n=Number.MAX_SAFE_INTEGER,s=e.map((i,u)=>le(i)?n:u),r=i=>{const u=s[t.get(i.name)];return u??-1},a=e.map((i,u)=>i.children.map(r).reduce((c,h)=>Math.max(c,h),s[u])),o=new Map;for(let i=0;i<e.length;++i){const u=a[i];if(u===n)continue;const c=e[i],h=e[u];o.has(h.name)||o.set(h.name,[]),o.get(h.name).push(c)}return o}const O1=new Set(["Switch","Merge","Enter","Exit","NextIteration","StatelessIf","StatelessWhile","if","While"]),F1=new Set(["NonMaxSuppressionV2","NonMaxSuppressionV3","NonMaxSuppressionV5","Where"]),C1=new Set(["HashTable","HashTableV2","LookupTableImport","LookupTableImportV2","LookupTableFind","LookupTableFindV2","LookupTableSize","LookupTableSizeV2"]);function le(e){return O1.has(e.op)}function B1(e){return F1.has(e.op)}function R1(e){return C1.has(e.op)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Pn{get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(t){const n=Object.keys(t).map(s=>t[s].map(r=>r.id));this._weightIds=[].concat(...n),this._weightMap=t}set resourceManager(t){this._resourceManager=t}get inputs(){return this._inputs.map(t=>({name:t.name,shape:t.attrParams.shape?t.attrParams.shape.value:void 0,dtype:t.attrParams.dtype?t.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(t=>({name:t.name,shape:t.attrParams.shape?t.attrParams.shape.value:void 0,dtype:t.attrParams.dtype?t.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(t=>t.signatureKey||t.name)}get outputNodes(){return this._outputs.map(t=>{const n=t.signatureKey||t.name;return t.defaultOutput?`${n}:${t.defaultOutput}`:n})}get functions(){return Object.keys(this._functions).reduce((t,n)=>(t[n]=this._functions[n].signature,t),{})}constructor(t,n){this.graph=t,this.parent=n,this.compiledMap=new Map,this.parseNodeNameCache=new Map,this._weightMap={},this.SEPARATOR=",",this._functions={},this._functionExecutorMap={},this.keepIntermediateTensors=!1,this._outputs=t.outputs,this._inputs=t.inputs,this._initNodes=t.initNodes,this._signature=t.signature,this._functions=t.functions,t.functions!=null&&Object.keys(t.functions).forEach(s=>{this._functionExecutorMap[s]=new Pn(t.functions[s],this)})}getCompilationKey(t,n){const s=t.map(a=>a.name).sort(),r=n.map(a=>a.name).sort();return s.join(this.SEPARATOR)+"--"+r.join(this.SEPARATOR)}compile(t,n){const s=Hr(t,n,this.weightMap,this._initNodes),{missingInputs:r,dynamicNode:a,syncInputs:o}=s;if(a!=null)throw new Error(`This execution contains the node '${a.name}', which has the dynamic op '${a.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${o}]`);if(r.length>0){const c=n.map(p=>p.name),h=Object.keys(t);throw new Error(`Cannot compute the outputs [${c}] from the provided inputs [${h}]. Missing the following inputs: [${r}]`)}const i=x1(this.graph,s),u=D1(i);return{orderedNodes:i,nodeLiveUntilMap:u}}cloneAndKeepTensor(t){if(t==null)return null;const n=t.clone();return Ct(n),n}cloneTensorList(t){return t?t.map(s=>this.cloneAndKeepTensor(s)):null}cloneTensorMap(t){return Object.fromEntries(Object.entries(t).map(([n,s])=>[n,this.cloneTensorList(s)]))}execute(t,n){this.disposeIntermediateTensors(),t=this.mapInputs(t);const s=Object.keys(t).sort();this.checkInputs(t),this.checkInputShapeAndType(t),n=this.mapOutputs(n),this.checkOutputs(n);const r=s.map(f=>this.graph.nodes[gt(f)[0]]),a=n.map(f=>gt(f)[0]),o=new Set(a);let i=a.map(f=>this.graph.nodes[f]);i.length===0&&(i=this._outputs);const u=this.getCompilationKey(r,i);let c=this.compiledMap.get(u);c==null&&(c=this.compile(t,i),this.compiledMap.set(u,c));try{this.keepIntermediateTensors=C().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(f){this.keepIntermediateTensors=!1,console.warn(f.message)}const h={},p={};return V(()=>{const f=new Kr(this.weightMap,h,p,this.functionExecutorMap,this.parseNodeNameCache),d=Object.assign({},this.weightMap);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap)),Object.keys(t).forEach(S=>{const[I,A]=gt(S,f),k=[];k[A]=t[S],d[I]=k,this.keepIntermediateTensors&&(this.clonedTensorsMap[I]=this.cloneTensorList(k))});const y=this.getFrozenTensorIds(d),{orderedNodes:T,nodeLiveUntilMap:N}=c;for(const S of T){if(d[S.name])continue;const I=Mr(S,d,f,this._resourceManager);if(te(I))throw new Error(`The execution of the op '${S.op}' returned a promise. Please use model.executeAsync() instead.`);d[S.name]=I,this.keepIntermediateTensors&&(this.clonedTensorsMap[S.name]=this.cloneTensorList(I)),this.checkTensorForDisposalWithNodeLiveUntilInfo(S,d,f,y,o,N.get(S.name))}return this.parent==null&&f.dispose(y),n.map(S=>at(S,d,f))})}getFrozenTensorIds(t){const n=[].concat.apply([],Object.keys(t).map(s=>t[s]).map(s=>s.map(r=>r.id)));return new Set(n)}checkTensorForDisposal(t,n,s,r,a,o,i){if(!(le(n)||o.has(t))){for(const u of s[t])u!=null&&(i[u.id]=(i[u.id]||0)+n.children.length);for(const u of n.inputs){if(le(u))continue;const c=Ur(u.name,s,r);if(c!=null)for(const h of c){if(!h||h.kept||a.has(h.id))continue;const p=i[h.id];p===1?(h.dispose(),delete i[h.id]):p!=null&&i[h.id]--}}}}checkTensorForDisposalWithNodeLiveUntilInfo(t,n,s,r,a,o){function i(u){return le(u)||a.has(u.name)}if(!(le(t)||o==null))for(const u of o){if(i(u))continue;const c=Ur(u.name,n,s);for(const h of c)!h||h.kept||r.has(h.id)||h.dispose()}}async executeAsync(t,n){return this._executeAsync(t,n)}disposeIntermediateTensors(){this.clonedTensorsMap&&(Object.values(this.clonedTensorsMap).forEach(t=>{for(const n of t)n&&!n.isDisposed&&n.dispose()}),this.clonedTensorsMap=null)}getIntermediateTensors(){return this.clonedTensorsMap}async _executeAsync(t,n,s=!1,r={},a={}){this.disposeIntermediateTensors(),s||(t=this.mapInputs(t),this.checkInputs(t),this.checkInputShapeAndType(t),n=this.mapOutputs(n),this.checkOutputs(n));try{this.keepIntermediateTensors=C().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(f){this.keepIntermediateTensors=!1,console.warn(f.message)}const o=new Kr(this.weightMap,r,a,this.functionExecutorMap,this.parseNodeNameCache);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap));const i=await this.executeWithControlFlow(t,o,n,s),u=n.map(f=>at(f,i,o)),c=u.map(f=>f.id),h=Object.keys(t).map(f=>t[f].id),p=new Set([...c,...h,...this.weightIds]);return Object.values(i).forEach(f=>{f.forEach(d=>{d&&!d.isDisposed&&!p.has(d.id)&&d.dispose()})}),this.parent==null&&o.dispose(p),u}async executeFunctionAsync(t,n,s){const r=t.reduce((a,o,i)=>(a[this.inputs[i].name]=o,a),{});return this._executeAsync(r,this.outputNodes,!0,n,s)}async executeWithControlFlow(t,n,s,r){const a=Object.keys(t),o=a.map(k=>this.graph.nodes[gt(k)[0]]),i=s.map(k=>gt(k)[0]),u=new Set(i);let c=i.map(k=>this.graph.nodes[k]);c.length===0&&(c=this._outputs);const{usedNodes:h,missingInputs:p,dynamicNode:f,syncInputs:d}=Hr(t,c,this.weightMap,this._initNodes),y=[...o,...this.graph.weights,...this._initNodes||[]].map(k=>({node:k,contexts:n.currentContext})),T=Object.assign({},this.weightMap);Object.keys(t).forEach(k=>{const[_,x]=gt(k),D=[];D[x]=t[k],T[_]=D});const N={},S=this.getFrozenTensorIds(T),I={};for(;y.length>0;){const k=this.processStack(o,y,n,T,I,S,u,N,h);await Promise.all(k)}f==null&&!r&&console.warn("This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.");const A=c.filter(k=>!le(k)&&!at(k.name,T,n)).map(k=>k.name);if(A.length>0){let k="";throw f!=null&&(k=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${d}]`),new Error(`Cannot compute the outputs [${A}] from the provided inputs [${a}]. Consider providing the following inputs: [${p}]. ${k}`)}return T}processStack(t,n,s,r,a,o,i,u,c){const h=[];for(;n.length>0;){const p=n.pop();s.currentContext=p.contexts;let f="";if(p.node.op==="Enter"&&l("isConstant",p.node,r,s)&&([f]=zt(p.node.name,s)),r[p.node.name]==null){const d=Mr(p.node,r,s,this._resourceManager);f||([f]=zt(p.node.name,s));const y=s.currentContext;te(d)?h.push(d.then(T=>(r[f]=T,this.keepIntermediateTensors&&(this.clonedTensorsMap[f]=this.cloneTensorList(T)),s.currentContext=y,this.checkTensorForDisposal(f,p.node,r,s,o,i,u),this.processChildNodes(p.node,n,s,r,a,c),T))):(r[f]=d,this.keepIntermediateTensors&&(this.clonedTensorsMap[f]=this.cloneTensorList(d)),this.checkTensorForDisposal(f,p.node,r,s,o,i,u),this.processChildNodes(p.node,n,s,r,a,c))}else this.processChildNodes(p.node,n,s,r,a,c)}return h}processChildNodes(t,n,s,r,a,o){t.children.forEach(i=>{const[u]=zt(i.name,s);a[u]||!o.has(i.name)||(i.op==="Merge"?i.inputNames.some(c=>!!at(c,r,s))&&(a[u]=!0,n.push({contexts:s.currentContext,node:i})):i.inputNames.every(c=>!!at(c,r,s))&&(a[u]=!0,n.push({contexts:s.currentContext,node:i})))})}dispose(){Object.keys(this.weightMap).forEach(t=>this.weightMap[t].forEach(n=>n.dispose()))}checkInputShapeAndType(t){Object.keys(t).forEach(n=>{const s=t[n],[r]=gt(n),a=this.graph.nodes[r];if(a.attrParams.shape&&a.attrParams.shape.value){const o=a.attrParams.shape.value,i=o.length===s.shape.length&&s.shape.every((u,c)=>o[c]===-1||o[c]===u);g(i,()=>`The shape of dict['${a.name}'] provided in model.execute(dict) must be [${o}], but was [${s.shape}]`)}a.attrParams.dtype&&a.attrParams.dtype.value&&g(s.dtype===a.attrParams.dtype.value,()=>`The dtype of dict['${a.name}'] provided in model.execute(dict) must be ${a.attrParams.dtype.value}, but was ${s.dtype}`)})}mapInputs(t){var n,s;const r={};for(const a in t){const o=(s=(n=this._signature)===null||n===void 0?void 0:n.inputs)===null||s===void 0?void 0:s[a];o!=null?r[o.name]=t[a]:r[a]=t[a]}return r}checkInputs(t){const n=Object.keys(t).filter(s=>{const[r]=gt(s);return this.graph.nodes[r]==null});if(n.length>0)throw new Error(`The dict provided in model.execute(dict) has keys: [${n}] that are not part of graph`)}mapOutputs(t){return t.map(n=>{var s,r;const a=(r=(s=this._signature)===null||s===void 0?void 0:s.outputs)===null||r===void 0?void 0:r[n];return a!=null?a.name:n},{})}checkOutputs(t){t.forEach(n=>{const[s]=gt(n);if(!this.graph.nodes[s])throw new Error(`The output '${n}' is not found in the graph`)})}}class P1{constructor(t={},n={}){this.hashTableNameToHandle=t,this.hashTableMap=n}addHashTable(t,n){this.hashTableNameToHandle[t]=n.handle,this.hashTableMap[n.id]=n}getHashTableHandleByName(t){return this.hashTableNameToHandle[t]}getHashTableById(t){return this.hashTableMap[t]}dispose(){for(const t in this.hashTableMap)this.hashTableMap[t].clearAndClose(),delete this.hashTableMap[t];for(const t in this.hashTableNameToHandle)this.hashTableNameToHandle[t].dispose(),delete this.hashTableNameToHandle[t]}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const L1="?tfjs-format=file",z1="model.json";class Wo{get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}constructor(t,n={},s=Po){this.modelUrl=t,this.loadOptions=n,this.version="n/a",this.io=s,n==null&&(this.loadOptions={}),this.resourceManager=new P1}findIOHandler(){const t=this.modelUrl;if(t.load!=null)this.handler=t;else if(this.loadOptions.requestInit!=null)this.handler=this.io.browserHTTPRequest(t,this.loadOptions);else{const n=this.io.getLoadHandlers(t,this.loadOptions);if(n.length===0)n.push(this.io.browserHTTPRequest(t,this.loadOptions));else if(n.length>1)throw new Error(`Found more than one (${n.length}) load handlers for URL '${[t]}'`);this.handler=n[0]}}load(){if(this.findIOHandler(),this.handler.load==null)throw new Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");const t=this.handler.load();return te(t)?t.then(n=>n.getWeightStream==null?this.loadSync(n):this.loadStreaming(n)):this.loadSync(t)}loadSync(t){const n=this.io.decodeWeights(t.weightData,t.weightSpecs);return this.loadWithWeightMap(t,n)}async loadStreaming(t){if(t.getWeightStream==null)throw new Error("Model artifacts missing streamWeights function");const n=await Ca(t.getWeightStream(),t.weightSpecs);return this.loadWithWeightMap(t,n)}loadWithWeightMap(t,n){this.artifacts=t;const s=this.artifacts.modelTopology;let r=this.artifacts.signature;if(this.artifacts.userDefinedMetadata!=null){const a=this.artifacts.userDefinedMetadata;a.signature!=null&&(r=a.signature),a.structuredOutputKeys!=null&&(this.structuredOutputKeys=a.structuredOutputKeys)}if(this.signature=r,this.version=`${s.versions.producer}.${s.versions.minConsumer}`,this.executor=new Pn(jr.Instance.transformGraph(s,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(n),this.executor.resourceManager=this.resourceManager,t.modelInitializer!=null&&t.modelInitializer.node!=null){const a=jr.Instance.transformGraph(t.modelInitializer);this.initializer=new Pn(a),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializerSignature=t.initializerSignature}return!0}async save(t,n){if(typeof t=="string"){const s=this.io.getSaveHandlers(t);if(s.length===0)throw new Error(`Cannot find any save handlers for URL '${t}'`);if(s.length>1)throw new Error(`Found more than one (${s.length}) save handlers for URL '${t}'`);t=s[0]}if(t.save==null)throw new Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return t.save(this.artifacts)}addStructuredOutputNames(t){if(this.structuredOutputKeys){const n=t instanceof et?[t]:t,s={};return n.forEach((r,a)=>s[this.structuredOutputKeys[a]]=r),s}return t}predict(t,n){const s=this.execute(t,this.outputNodes);return this.addStructuredOutputNames(s)}async predictAsync(t,n){const s=await this.executeAsync(t,this.outputNodes);return this.addStructuredOutputNames(s)}normalizeInputs(t){var n;if(!(t instanceof et)&&!Array.isArray(t)){const a=(n=this.signature)===null||n===void 0?void 0:n.inputs;if(a!=null)for(const o in a){const i=a[o];i.resourceId!=null&&(t[o]=this.resourceIdToCapturedInput[i.resourceId])}return t}t=Array.isArray(t)?t:[t];const s=Object.keys(this.resourceIdToCapturedInput).length;if(t.length+s!==this.inputNodes.length)throw new Error(`Input tensor count mismatch, the graph model has ${this.inputNodes.length-s} non-resource placeholders, while there are ${t.length} input tensors provided.`);let r=0;return this.inputNodes.reduce((a,o)=>{var i,u,c;const h=(c=(u=(i=this.signature)===null||i===void 0?void 0:i.inputs)===null||u===void 0?void 0:u[o])===null||c===void 0?void 0:c.resourceId;return h!=null?a[o]=this.resourceIdToCapturedInput[h]:a[o]=t[r++],a},{})}normalizeOutputs(t){return t=t||this.outputNodes,Array.isArray(t)?t:[t]}executeInitializerGraph(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.execute({},[]):this.initializer.execute({},Object.keys(this.initializerSignature.outputs))}async executeInitializerGraphAsync(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.executeAsync({},[]):this.initializer.executeAsync({},Object.keys(this.initializerSignature.outputs))}setResourceIdToCapturedInput(t){if(this.resourceIdToCapturedInput={},this.initializerSignature){const n=this.initializerSignature.outputs,s=Object.keys(n);for(let r=0;r<s.length;r++){const a=s[r],o=n[a];this.resourceIdToCapturedInput[o.resourceId]=t[r]}}}execute(t,n){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(this.executeInitializerGraph()),t=this.normalizeInputs(t),n=this.normalizeOutputs(n);const s=this.executor.execute(t,n);return s.length>1?s:s[0]}async executeAsync(t,n){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(await this.executeInitializerGraphAsync()),t=this.normalizeInputs(t),n=this.normalizeOutputs(n);const s=await this.executor.executeAsync(t,n);return s.length>1?s:s[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(t){return Object.keys(t).reduce((n,s)=>(n[s]=[t[s]],n),{})}dispose(){this.executor.dispose(),this.initializer&&(this.initializer.dispose(),this.resourceIdToCapturedInput&&pt(this.resourceIdToCapturedInput)),this.resourceManager.dispose()}}async function JN(e,t={},n=Po){if(e==null)throw new Error("modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model");t==null&&(t={}),t.fromTFHub&&typeof e=="string"&&(e=V1(e));const s=new Wo(e,t,n);return await s.load(),s}function YN(e){if(e==null)throw new Error("modelUrl in loadGraphModelSync() cannot be null. Please provide model artifacts or an IOHandler that loads the model");let t;if(e instanceof Array){const[s,r]=e;if(!s)throw new Error("modelJSON must be the first element of the array");if(!r||!(r instanceof ArrayBuffer))throw new Error("An ArrayBuffer of weights must be the second element of the array");if(!("modelTopology"in s))throw new Error("Model JSON is missing 'modelTopology'");if(!("weightsManifest"in s))throw new Error("Model JSON is missing 'weightsManifest'");const a=In(s.weightsManifest),o=js(s,a,r);t=Bn(o)}else if("load"in e)t=e;else if("modelTopology"in e&&"weightSpecs"in e&&"weightData"in e)t=Bn(e);else throw new Error("Unknown model format");const n=new Wo(t);return n.load(),n}function V1(e){return e.endsWith("/")||(e=e+"/"),`${e}${z1}${L1}`}export{H1 as $,Ti as A,Ri as B,ua as C,Ui as D,ji as E,Wi as F,qi as G,Gi as H,Mi as I,Ki as J,Hi as K,Xi as L,M1 as M,Ji as N,Yi as O,Zi as P,eu as Q,Qi as R,tu as S,j1 as T,nu as U,su as V,ru as W,au as X,ou as Y,iu as Z,uu as _,Si as a,Qu as a$,K1 as a0,ca as a1,w as a2,aa as a3,lu as a4,hu as a5,X1 as a6,di as a7,fu as a8,pu as a9,Ou as aA,Fu as aB,Cu as aC,Bu as aD,Ru as aE,Pu as aF,Y1 as aG,Lu as aH,zu as aI,Vu as aJ,J1 as aK,Z1 as aL,tN as aM,ju as aN,qu as aO,Gu as aP,nN as aQ,eN as aR,Mu as aS,Wu as aT,Ku as aU,Hu as aV,Xu as aW,Ju as aX,Yu as aY,zw as aZ,Zu as a_,mu as aa,du as ab,gu as ac,yu as ad,bu as ae,wu as af,Nu as ag,Tu as ah,yr as ai,Su as aj,wr as ak,Nr as al,ku as am,$u as an,Wo as ao,Eu as ap,vu as aq,_u as ar,la as as,xu as at,Iu as au,Au as av,Du as aw,Jo as ax,Uu as ay,Q1 as az,$i as b,Jc as b$,tc as b0,nc as b1,sc as b2,rc as b3,ec as b4,Da as b5,oc as b6,ac as b7,_e as b8,ic as b9,Oo as bA,vc as bB,xc as bC,Ic as bD,Ac as bE,Ao as bF,Ht as bG,Bc as bH,Cc as bI,Oc as bJ,Fc as bK,Dc as bL,Uc as bM,Rc as bN,zc as bO,jc as bP,Wc as bQ,qc as bR,Gc as bS,Mc as bT,Vc as bU,Pc as bV,oN as bW,Kc as bX,Hc as bY,ul as bZ,Xc as b_,uc as ba,sN as bb,cc as bc,lc as bd,hc as be,Vw as bf,pc as bg,fc as bh,mc as bi,dc as bj,Ir as bk,gc as bl,cu as bm,yc as bn,ht as bo,bc as bp,Sc as bq,wc as br,Tc as bs,aN as bt,Nc as bu,rN as bv,$c as bw,cl as bx,kc as by,Ec as bz,Bw as c,Ft as c$,Yc as c0,Zc as c1,Qc as c2,Lc as c3,tl as c4,el as c5,et as c6,is as c7,_c as c8,ha as c9,lt as cA,Hh as cB,Jh as cC,Zh as cD,Ma as cE,cp as cF,Qa as cG,Ul as cH,gp as cI,jn as cJ,Tp as cK,$p as cL,Ep as cM,Ka as cN,Ha as cO,xp as cP,by as cQ,Ap as cR,Sn as cS,BN as cT,KN as cU,e0 as cV,jt as cW,En as cX,ey as cY,tt as cZ,Fp as c_,nl as ca,sl as cb,Yn as cc,rl as cd,al as ce,ol as cf,iN as cg,xn as ch,il as ci,br as cj,Nt as ck,Ah as cl,Oh as cm,F as cn,Ch as co,Rh as cp,Lh as cq,cr,Vh as cs,jh as ct,Pt as cu,qh as cv,Mh as cw,g as cx,nt as cy,PN as cz,Rw as d,yN as d$,Zo as d0,Bp as d1,Yt as d2,Cf as d3,ee as d4,Un as d5,tp as d6,Ga as d7,FN as d8,RN as d9,bl as dA,uf as dB,lf as dC,qt as dD,os as dE,Oa as dF,pf as dG,NN as dH,mf as dI,qs as dJ,zy as dK,Uy as dL,XN as dM,gN as dN,yf as dO,wf as dP,wN as dQ,pt as dR,TN as dS,K as dT,Ef as dU,_f as dV,Dy as dW,Oe as dX,ae as dY,Za as dZ,bN as d_,Qh as da,CN as db,pn as dc,ct as dd,Pp as de,zp as df,Up as dg,Wp as dh,Hl as di,Mp as dj,By as dk,Xa as dl,Wn as dm,Py as dn,Xp as dp,Qp as dq,Yp as dr,ef as ds,hi as dt,ap as du,m as dv,pN as dw,sf as dx,af as dy,or as dz,Pw as e,Pl as e$,_o as e0,zs as e1,jl as e2,SN as e3,Df as e4,C as e5,Ya as e6,Ff as e7,Wf as e8,be as e9,vn as eA,ss as eB,nh as eC,Ja as eD,eh as eE,Vs as eF,Zr as eG,zN as eH,UN as eI,jN as eJ,Mn as eK,ro as eL,ci as eM,Cn as eN,Kn as eO,xw as eP,Fy as eQ,mi as eR,hn as eS,ui as eT,se as eU,Po as eV,To as eW,Ia as eX,nm as eY,rm as eZ,Le as e_,Kt as ea,qn as eb,Kf as ec,eo as ed,Nl as ee,nr as ef,dn as eg,IN as eh,AN as ei,yl as ej,Ue as ek,no as el,qa as em,m0 as en,My as eo,so as ep,xy as eq,Cs as er,LN as es,Vl as et,Nf as eu,ur as ev,ir as ew,Sa as ex,Tr as ey,VN as ez,Lw as f,b as f$,om as f0,sa as f1,te as f2,Ln as f3,Dl as f4,St as f5,Ct as f6,ao as f7,ws as f8,Gs as f9,Fm as fA,Bm as fB,po as fC,On as fD,$N as fE,Lm as fF,bs as fG,Fn as fH,Um as fI,Wm as fJ,Gm as fK,Ty as fL,v as fM,Km as fN,Xm as fO,li as fP,Qo as fQ,Rt as fR,db as fS,gb as fT,yb as fU,Gn as fV,fo as fW,tn as fX,Zm as fY,fe as fZ,td as f_,Iw as fa,lm as fb,JN as fc,YN as fd,Yw as fe,fi as ff,pm as fg,uN as fh,an as fi,oo as fj,wm as fk,Sm as fl,uo as fm,Dn as fn,co as fo,lo as fp,xm as fq,Aw as fr,Am as fs,Bs as ft,pi as fu,zn as fv,U as fw,Gy as fx,Pe as fy,ho as fz,ia as g,vg as g$,nd as g0,gn as g1,ad as g2,id as g3,cd as g4,hd as g5,ln as g6,gd as g7,rn as g8,go as g9,Db as gA,Fb as gB,we as gC,ig as gD,cg as gE,hg as gF,fg as gG,sr as gH,Je as gI,No as gJ,gg as gK,z as gL,$y as gM,GN as gN,Ms as gO,Qs as gP,bg as gQ,Ng as gR,MN as gS,vN as gT,ON as gU,Sg as gV,lr as gW,Xr as gX,Re as gY,kg as gZ,_w as g_,Wa as ga,wd as gb,kN as gc,Td as gd,$d as ge,Ed as gf,_d as gg,Kd as gh,bo as gi,Jd as gj,er as gk,Qd as gl,on as gm,_N as gn,un as go,ng as gp,DN as gq,Do as gr,cN as gs,ll as gt,HN as gu,Xn as gv,wo as gw,xN as gx,ii as gy,$ as gz,ki as h,dm as h$,xg as h0,Z as h1,ai as h2,q as h3,Ag as h4,Og as h5,Cg as h6,Rg as h7,Lg as h8,io as h9,Zg as hA,Qg as hB,ty as hC,sy as hD,dN as hE,V as hF,Ye as hG,EN as hH,me as hI,Vn as hJ,ay as hK,Ns as hL,iy as hM,An as hN,cy as hO,hN as hP,lN as hQ,hy as hR,ve as hS,$a as hT,fy as hU,fN as hV,ar as hW,Eo as hX,WN as hY,qN as hZ,my as h_,mo as ha,Dw as hb,vy as hc,vw as hd,cn as he,Wt as hf,At as hg,So as hh,rr as hi,Yr as hj,Gt as hk,$o as hl,Xg as hm,We as hn,Ow as ho,P as hp,M as hq,ti as hr,mN as hs,kn as ht,Yg as hu,ys as hv,Bt as hw,xt as hx,Ze as hy,ko as hz,Ei as i,ue as i0,Zt as i1,vo as i2,dy as i3,qe as i4,Tt as i5,vi as j,_i as k,xi as l,Ii as m,Ai as n,Di as o,Fi as p,Oi as q,Ci as r,Bi as s,q1 as t,W1 as u,Pi as v,Li as w,zi as x,Vi as y,G1 as z};
