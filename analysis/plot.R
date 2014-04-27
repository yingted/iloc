root <- '.'
# files <- c('c', 'l', 'r')
files <- c('normal/c', 'normal/l', 'normal/r', 'center/c', 'center/l', 'center/r')

colours <- 1:length(files)
data <- NULL
for (i in colours) {
	cur <- read.table(paste(root, files[i], sep='/'))
	cur$s <- i
	data <- rbind(data, cur)
}

dev.prev()
plot(V2~V1, data, col=data$s, xlim=c(0, 1), ylim=c(0, 1), pch=19, cex=.1)
x <- c(0, 1)
for (d in -10:10/10) {
	lines(x, x + d, col=length(files) + 1)
}
legend('topleft', files, col=colours, lty=1)
dev.new()
plot.new()

require(lattice)
densityplot(~(V2 - V1), data=data, groups=s, par.settings = list(superpose.line = list(col = colours)))
legend('topleft', files, col=colours, lty=1)
