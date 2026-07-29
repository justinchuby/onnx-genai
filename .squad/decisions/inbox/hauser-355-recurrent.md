### 2026-07-29: Share recurrent output geometry across RNN, GRU, and LSTM
**By:** Hauser
**What:** RNN, GRU, and LSTM shape inference uses one helper that propagates X's sequence and batch dimensions, derives the direction count and hidden size from attributes, and emits only the declared Y/Y_h/Y_c outputs. Registrations at opsets 1 and 14 select whether the helper ignores or honors the layout attribute.
**Why:** The three operators have the same tensor geometry except that LSTM has an additional cell-state output. Opset 14 is the layout boundary: earlier schemas are always sequence-major, while layout=1 from opset 14 onward moves batch ahead of sequence and direction in every output.
