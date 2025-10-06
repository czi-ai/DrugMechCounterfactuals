Contents of Data/Sessions/Variances
-----------------------------------

This dir contains options files for use with commands in the Python package `drugmechcf.exp.cfvariances`.
This package runs multiple sessions of a set of queries against the specified LLM models, to compute mean
accuracy and confidence interval for models with stochastically variable responses.

To get help on the commands supported in this package, execute the following in a shell:

```
(dmcf) $ cd $PROJDIR/src
(dmcf) $ python -m drugmechcf.exp.cfvariances -h
```

The following options files were used to derive the metrics described in our paper:

opt_add_closed.json	... Add-Link, closed world.
opt_add_open.json	... Add-Link, open world.

opt_change_closed.json	... Invert-Link, closed world.
opt_change_open.json	... Invert-Link, open world.

opt_delete_closed.json	... Delete-Link, closed world.
opt_delete_open.json	... Delete-Link, open world.
