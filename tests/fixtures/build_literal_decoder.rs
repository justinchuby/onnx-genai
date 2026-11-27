use std::path::Path;
use serde_json::{Value,json};
fn copy(source:&Path,target:&Path) {
    std::fs::create_dir_all(target).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {let entry=entry.unwrap();let dest=target.join(entry.file_name());if entry.file_type().unwrap().is_dir() {copy(&entry.path(),&dest);} else {std::fs::copy(entry.path(),dest).unwrap();}}
}
pub fn build(root:&Path,fixtures:&Path,split:bool) {
    copy(&fixtures.join("tiny-llm"),root);
    let path=root.join("tokenizer.json");let mut tokenizer:Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let vocab=tokenizer["model"]["vocab"].as_object_mut().unwrap();
    for (id,text) in if split {[(22,"@a@"),(23,"@b@"),(24,"@bad@")] } else {[(22,"@source@"),(23,"@bad@"),(24,"@other@")] } {
        let old=vocab.iter().find(|(_,v)|v.as_u64()==Some(id)).unwrap().0.clone();vocab.remove(&old);assert!(vocab.insert(text.into(),json!(id)).is_none());
    }
    tokenizer["decoder"]=json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Replace","pattern":{"String":if split {"@a@@b@"} else {"@source@"}},"content":"7"}]});
    tokenizer["added_tokens"]=json!([{"id":3,"content":"<eos>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}]);
    std::fs::write(path,serde_json::to_vec(&tokenizer).unwrap()).unwrap();
}

/// Input-dependent analytical head: prompt ->22,22->23,23->24.
/// The retained cache/transformer graph remains real Native execution.
pub fn invalid_group(root:&Path,fixtures:&Path,byte_level:bool) {
    build(root,fixtures,false);
    let path=root.join("tokenizer.json");let mut tokenizer:Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let vocab=tokenizer["model"]["vocab"].as_object_mut().unwrap();
    for (id,text) in if byte_level {[(22,"Ã"),(23,"©"),(24,"ÿ")] } else {[(22,"<0xC3>"),(23,"<0xA9>"),(24,"<0xFF>")] } {
        let old=vocab.iter().find(|(_,v)|v.as_u64()==Some(id)).unwrap().0.clone();vocab.remove(&old);assert!(vocab.insert(text.into(),json!(id)).is_none());
    }
    let width=vocab.len();
    tokenizer["decoder"]=json!({"type":"Sequence","decoders":[if byte_level {json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false})} else {json!({"type":"ByteFallback"})},{"type":"Fuse"},{"type":"Replace","pattern":{"String":if byte_level {"é�"} else {"���"}},"content":"7"}]});
    std::fs::write(path,serde_json::to_vec(&tokenizer).unwrap()).unwrap();
    let model=root.join("model.onnx.textproto");let source=std::fs::read_to_string(&model).unwrap();assert!(source.contains("output: \"logits\""));
    let source=source.replace("output: \"logits\"","output: \"original_unused_logits\"");
    let mut values=String::new();for row in 0..width {let target=match row {22=>23,23=>24,_=>22};for col in 0..width {values.push_str(if col==target {" float_data: 14"} else {" float_data: -14"});}}
    let head=format!("  initializer {{name:\"literal_transition\" dims:{width} dims:{width} data_type:1 {values}}}\n  node {{input:\"literal_transition\" input:\"input_ids\" output:\"logits\" name:\"literal_transition_head\" op_type:\"Gather\"}}\n");
    let anchor="  name: \"main_graph\"";assert_eq!(source.matches(anchor).count(),1);std::fs::write(model,source.replacen(anchor,&format!("{head}{anchor}"),1)).unwrap();
}
