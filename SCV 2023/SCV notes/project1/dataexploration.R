### Import data

# https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set

library("readxl")
data <- read_excel('data.xlsx')
colnames(data) <- c("No","date","age", "dist_MRT", "conv_stores", "lat", "long", "price")

str(data)
head(data)

data <- subset(data, select = -c(No))
ggpairs((data))

"
Just looking at the data, there seem to be a nice distribution for the variables overall.
Looking at the distribution of the variable `distance` and the response-variable `price`, we see that it is heavily right-skewed.
To fix this we log-transform this `distance`. The variable `long` also seems to be skewed, but doesn't change much with 
a log-transformation, so we keep it as it is.
"

data$dist_MRT <- log10(data$dist_MRT)
data$price <- log10(data$price)

"
We know want to explore the correlation in the variables.
"

library(corrplot)
par(mfrow=c(1,1))
M <- cor(data)
corrplot(cor(data), method="number")

"
We see that `price` has a strong correlation with several of the variables,
but there is also correlation between the variables themselves.
To explore this further we want to do some regression plots on the variables.
Firstly we want to plot each variable with the response-variable `price`.
"

fig1 <- ggplot(data, aes(date, price)) + geom_point() + stat_smooth(method="lm")
fig2 <- ggplot(data, aes(data$age, data$price)) + geom_point() + stat_smooth(method="lm")
fig3 <- ggplot(data, aes(data$dist_MRT, data$price)) + geom_point() + stat_smooth(method="lm")
fig4 <- ggplot(data, aes(data$conv_stores, data$price)) + geom_point() + stat_smooth(method="lm")
fig5 <- ggplot(data, aes(data$lat, data$price)) + geom_point() + stat_smooth(method="lm")
fig6 <- ggplot(data, aes(data$long, data$price)) + geom_point() + stat_smooth(method="lm")

library(ggpubr)
ggarrange(fig1, fig2, fig3, fig4, fig5, fig6)

"
Looking at the regression plots, we see that `price` has a high correletation with `dist_MRT` `conv_stores`, `lat` and `long`, which we also saw in the correlation plot.
This makes sence though, since number of convenience stores in the area and the distance to the metro,
is dependent on where the house is located (latitude and longitude).

For this reason we plotted `dist_MRT`, `conv_stores` and also `price` with the two variables `lat` and `long`.
The plot for `conv_stores` with `lat` and `long` is shown below.
"

# Put plot here

"
All three plots show the same thing. There seems to be a city centre, where a lot of houses are located.
Here the distance to the metro is low, the number of convenience stores is high and the prices are high compared to the other locations.
"



