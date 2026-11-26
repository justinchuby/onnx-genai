import os,pathlib,re,subprocess,sys,json,hashlib
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');out=pathlib.Path('/tmp/nxrt-prefetch-groups-validation');out.mkdir(exist_ok=True)
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
