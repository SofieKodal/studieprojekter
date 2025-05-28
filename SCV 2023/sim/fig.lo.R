## plots for medium-dimensional simulation, s=5
library(bestsubset)
n = 100; p = 10
file.list = system(paste0("ls ","rds/lo/*.rds"),intern=TRUE)
method.nums = c(2,1,3,4,5)
method.names = c("Forward stepwise","Lasso","Relaxed lasso", "Elastic net", "Adaptive lasso")


plot.from.file(file.list, what="error", rel.to=NULL, tuning="val",
               method.nums=method.nums, method.names=method.names,
               legend.pos="bottom", make.pdf=TRUE, fig.dir="fig/lo",
               file.name="lo.err")

plot.from.file(file.list, what="prop", rel.to=NULL, tuning="val",
               method.nums=method.nums, method.names=method.names,
               legend.pos="bottom", make.pdf=TRUE, fig.dir="fig/lo",
               file.name="lo.prop")

plot.from.file(file.list, what="nonzero", rel.to=NULL, tuning="val",
               method.nums=method.nums, method.names=method.names,
               legend.pos="bottom", make.pdf=TRUE, fig.dir="fig/lo",
               file.name="lo.nzs")

plot.from.file(file.list, what="F", rel.to=NULL, tuning="val",
               method.nums=method.nums, method.names=method.names,
               legend.pos="bottom", make.pdf=TRUE, fig.dir="fig/lo",
               file.name="lo.F")