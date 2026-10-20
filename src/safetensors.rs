use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn expect_shape(&self, name: &str, shape: &[usize]) -> Result<()> {
        if self.shape != shape {
            return Err(Error(format!(
                "tensor {name}: expected shape {shape:?}, got {:?}",
                self.shape
            )));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct SafeTensors {
    tensors: HashMap<String, Tensor>,
}

impl SafeTensors {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error(format!("cannot read {}: {e}", path.display())))?;
        if bytes.len() < 8 {
            return Err(Error(
                "safetensors file is shorter than the header length".into(),
            ));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let data_start = 8usize
            .checked_add(header_len)
            .ok_or_else(|| Error("invalid safetensors header length".into()))?;
        if data_start > bytes.len() {
            return Err(Error("safetensors header extends past end of file".into()));
        }
        let header: Value = serde_json::from_slice(&bytes[8..data_start])
            .map_err(|e| Error(format!("invalid safetensors header JSON: {e}")))?;
        let object = header
            .as_object()
            .ok_or_else(|| Error("safetensors header is not a JSON object".into()))?;
        let payload = &bytes[data_start..];
        let mut tensors = HashMap::with_capacity(object.len());
        for (name, info) in object {
            if name == "__metadata__" {
                continue;
            }
            let info = info
                .as_object()
                .ok_or_else(|| Error(format!("tensor {name}: header entry is not an object")))?;
            let dtype = info
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| Error(format!("tensor {name}: missing dtype")))?;
            if dtype != "F32" {
                return Err(Error(format!(
                    "tensor {name}: unsupported dtype {dtype}; this engine supports F32 checkpoints"
                )));
            }
            let shape: Vec<usize> = info
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| Error(format!("tensor {name}: missing shape")))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|x| x as usize)
                        .ok_or_else(|| Error(format!("tensor {name}: invalid shape entry")))
                })
                .collect::<Result<_>>()?;
            let offsets = info
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| Error(format!("tensor {name}: missing data_offsets")))?;
            if offsets.len() != 2 {
                return Err(Error(format!(
                    "tensor {name}: data_offsets must have two entries"
                )));
            }
            let start = offsets[0]
                .as_u64()
                .ok_or_else(|| Error(format!("tensor {name}: invalid data offset")))?
                as usize;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| Error(format!("tensor {name}: invalid data offset")))?
                as usize;
            let numel: usize = shape.iter().product();
            if end < start || end - start != numel * 4 {
                return Err(Error(format!(
                    "tensor {name}: byte range {start}..{end} does not match {numel} f32 values"
                )));
            }
            if end > payload.len() {
                return Err(Error(format!(
                    "tensor {name}: data extends past end of file"
                )));
            }
            let mut data = Vec::with_capacity(numel);
            for chunk in payload[start..end].chunks_exact(4) {
                data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            tensors.insert(name.clone(), Tensor { shape, data });
        }
        Ok(Self { tensors })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error(format!("checkpoint is missing tensor {name:?}")))
    }

    pub fn maybe(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }
}
