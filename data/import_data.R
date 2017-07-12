library(tidyverse)
library(earth)

data(trees)
data(ozone1)

trees %>% write_csv("trees.csv")
ozone1 %>% write_csv("ozone.csv")

# http://www.milbo.users.sonic.net/earth/earth-times.html
robotArm <- function(x) { # lifted from Friedman's Fast MARS paper
  x. <- with(x, l1 * cos(theta1) - l2 * cos(theta1 + theta2) * cos(phi))
  y  <- with(x, l1 * sin(theta1) - l2 * sin(theta1 + theta2) * cos(phi))
  z  <- with(x, l2 * sin(theta2) * sin(phi))
  sqrt(x.^2 + y^2 + z^2)
}
set.seed(1)   # for reproducibility
ncases <- 3000
l1     <- runif(ncases, 0, 1)
l2     <- runif(ncases, 0, 1)
theta1 <- runif(ncases, 0, 2 * pi)
theta2 <- runif(ncases, 0, 2 * pi)
phi    <- runif(ncases, -pi/2, pi/2)
x <- cbind(l1, l2, theta1, theta2, phi)
for (i in 1:25) {# 25 dummy vars, so 30 vars in total
  x <- cbind(x, runif(ncases, 0, 1))
}

robot_arm <- data.frame(x)
robot_arm$response <- robotArm(robot_arm)

robot_arm %>% write_csv("robotarm.csv")
