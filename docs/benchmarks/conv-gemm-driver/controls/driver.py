from pathlib import Path
import subprocess,json,hashlib,os
root=Path('/tmp/nxrt-qmoe-workspace');p=root/'crates/onnx-runtime-ep-cpu/src/kernels/conv_ref.rs';original=p.read_bytes();source=original.decode();base=Path('/tmp/nxrt-conv-driver-base.rs').read_text();out=Path('/tmp/nxrt-conv-driver-controls');out.mkdir(exist_ok=True)
ns={};exec(Path('/tmp/nxrt-prefetch-groups-check.py').read_text().split('commands=')[0],ns);env=ns['env']|{'RAYON_NUM_THREADS':'1'};records=[]
sha=lambda data:hashlib.sha256(data).hexdigest()
def run(name,body,args,extra={},success=True):
 p.write_text(body);(out/(name+'.rs')).write_text(body)
 cmd=['cargo',*args]
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=root,env=env|extra,stdout=f,stderr=subprocess.STDOUT)
 log=(out/(name+'.log')).read_text();records.append({'name':name,'command':cmd,'extra_env':extra,'exit':r.returncode,'source_sha256':sha(body.encode()),'log_sha256':sha(log.encode())});(out/'controls.json').write_text(json.dumps(records,indent=2));print(name,r.returncode,log[-2000:],flush=True)
 if success:assert r.returncode==0 and '1 passed; 0 failed' in log
 else:assert r.returncode!=0 and '0 passed; 1 failed' in log and 'error[E' not in log
try:
 run('historical-allocation',base,['test','--offline','-p','onnx-runtime-ep-cpu','--test','conv_im2col_allocation','--','--nocapture'],success=False)
 run('eager-col',source.replace('''    let mut col = if is_1x1 {
        Vec::new()
    } else {
        vec![0.0f32; k * n]
    };''','''    let mut col = vec![0.0f32; k * n];'''),['test','--offline','-p','onnx-runtime-ep-cpu','--test','conv_im2col_allocation','--','--nocapture'],success=False)
 a=base.index('fn im2col_gemm_execute(');b=base.index('fn im2col(',a);old1=base[a:b]
 a=base.index('fn grouped_im2col_gemm_execute(');b=base.index('\n// ═══',a);old2=base[a:b]
 old=(old1+old2).replace('super::matmul::','crate::kernels::matmul::')
 a=source.index('    #[test]\n    fn gemm_driver_matches_exact_scalar_values_across_groups_batches_and_empty_axes()');b=source.index('    #[test]\n    fn conv_2d_bias_stride_and_explicit_padding()',a)
 test=source[a:b].replace('gemm_driver_matches_exact_scalar_values_across_groups_batches_and_empty_axes','single_image_driver_bits_match_historical_implementations').replace('let expected = scalar_ref_execute(', 'let expected = historical_driver(')
 # Replace only fixture generation expressions; retain all 540 matrix cases.
 test=test.replace('(i as i32 % 13 - 6) as f32 * 0.25', 'sample(i, 13)')
 test=test.replace('(i as i32 % 7 - 3) as f32 * 0.5', 'sample(i, 7)')
 test=test.replace('(i as i32 % 5 - 2) as f32 * 0.125', 'sample(i, 5)')
 helper='''
fn sample(i: usize, seed: u64) -> f32 {
    let x = (i as u64).wrapping_add(seed).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = ((x >> 32) as u32 & 0x807fffff) | (((x >> 17) as u32 % 24 + 115) << 23);
    f32::from_bits(bits)
}
#[allow(clippy::too_many_arguments)]
fn historical_driver(x: &[f32], w: &[f32], bias: Option<&[f32]>, xs: &[usize], ws: &[usize], ys: &[usize], group: usize, strides: &[usize], dilations: &[usize], pads: &[usize], relu: bool) -> Result<Vec<f32>> {
 if group == 1 { im2col_gemm_execute(x,w,bias,xs,ws,ys,strides,dilations,pads,relu) }
 else { grouped_im2col_gemm_execute(x,w,bias,xs,ws,ys,group,strides,dilations,pads,relu) }
}
'''
 # New driver's scalar call was non-Result, whereas historical driver returns Result.
 test=test.replace('''                                );
                                assert_eq!(actual.len(),''','''                                ).unwrap();
                                assert_eq!(actual.len(),''')
 assert 'historical_driver(' in test and 'sample(i, 13)' in test and ').unwrap();' in test
 test=test.replace('assert_eq!(cases, 540,', 'println!("historical/new comparison executed {cases} geometries"); assert_eq!(cases, 540,')
 body=source+'\n#[cfg(test)]\nmod historical_conv_driver_probe {\nuse super::*;\n'+old+helper+test+'\n}\n'
 args=['test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::conv::historical_conv_driver_probe::single_image_driver_bits_match_historical_implementations','--','--exact','--nocapture']
 run('same-image',body,args)
 run('same-image-generic',body,args,{'NXRT_CPU_GEMM_BACKEND':'generic'})
finally:
 p.write_bytes(original);assert p.read_bytes()==original
 (out/'restoration.json').write_text(json.dumps({'restored_sha256':sha(p.read_bytes())},indent=2))
