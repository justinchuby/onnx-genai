//! Test-only observations of actual expert projection boundaries.
//! No observer state or calls exist in non-test builds.

use std::cell::RefCell;

#[derive(Clone, Debug)]
pub(crate) struct Projection {
    pub(crate) name: &'static str,
    pub(crate) input: Vec<f32>,
    pub(crate) output: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) rows: usize,
}

thread_local! {
    static RECORDS: RefCell<Option<Vec<Projection>>> = const { RefCell::new(None) };
}

pub(crate) struct Trace(std::marker::PhantomData<std::rc::Rc<()>>);
impl Trace {
    pub(crate) fn start() -> Self {
        RECORDS.with(|records| {
            let mut records = records.borrow_mut();
            assert!(records.is_none(), "expert numerical traces must not nest");
            *records = Some(Vec::new());
        });
        Self(std::marker::PhantomData)
    }
    pub(crate) fn finish(self) -> Vec<Projection> {
        RECORDS.with(|records| records.borrow_mut().take().expect("active numerical trace"))
    }
}
impl Drop for Trace {
    fn drop(&mut self) {
        RECORDS.with(|records| *records.borrow_mut() = None);
    }
}

pub(super) fn record(
    name: &'static str,
    input: &[f32],
    rows: usize,
    bias: Option<&[f32]>,
    output: &[f32],
) {
    RECORDS.with(|records| {
        if let Some(records) = records.borrow_mut().as_mut() {
            records.push(Projection {
                name,
                input: input.to_vec(),
                output: output.to_vec(),
                bias: bias.map(<[f32]>::to_vec),
                rows,
            });
        }
    });
}

#[test]
fn scoped_trace_is_thread_local_and_clears_after_unwind() {
    let trace = Trace::start();
    std::thread::spawn(|| record("other-thread", &[9.], 1, None, &[9.]))
        .join()
        .unwrap();
    record("fc1", &[1.], 1, Some(&[2.]), &[3.]);
    let records = trace.finish();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].name, "fc1");
    assert!(
        std::panic::catch_unwind(|| {
            let _trace = Trace::start();
            panic!("observer scope failed");
        })
        .is_err()
    );
    assert!(Trace::start().finish().is_empty());
}
