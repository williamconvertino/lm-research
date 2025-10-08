Transformer - Vanilla Transformer
FST - Vanilla FST
FST Clean - FST with the MLP layers isolated to each deal with only a single "f" output
FST Clean (resid=True) - Same as "Clean" but allows for the (pre-MLP) "f" to be included in the skip connection
FST Clean Sparse - Same as "Clean" but with L1 norm on the "f" MLPs
FST Feature First - FST but with all the feature layers first
FST 32 - FST with 2 feature layers per ICL layer
FST 23 - FST with 2 ICL layers per feature layer