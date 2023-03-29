## ----setup, include=FALSE--------------------------------------------------
library(tidyverse)
library(fpp3)
library(patchwork)
library(gganimate)
library(kableExtra)

austa <- as_tsibble(fpp2::austa) %>%
  rename(Year = index, Visitors = value)
melsyd <- tsibbledata::ansett %>%
  filter(Airports == "MEL-SYD")
global_economy <- global_economy %>%
  select(Year, Country, GDP, Imports, Exports, Population)
gafa_stock <- gafa_stock %>%
  select(Symbol, Date, Close, Volume)


## ---- echo = TRUE----------------------------------------------------------
global_economy


## ---- echo = TRUE----------------------------------------------------------
gafa_stock


## ----tstable, cache=TRUE---------------------------------------------------
mydata <- tsibble(
    year = 2012:2016,
    y = c(123, 39, 78, 52, 110),
    index = year
)
mydata


## ----astsibble, cache=TRUE-------------------------------------------------
mydata <- tibble(
    year = 2012:2016,
    y = c(123, 39, 78, 52, 110)
  ) %>%
  as_tsibble(index = year)
mydata


## ----tstablemonth, echo=FALSE----------------------------------------------
z <- tibble(Month = paste(2019, month.abb[1:5]), Observation = c(50, 23, 34, 30, 25))


## ----tstablemonth2---------------------------------------------------------
z


## ----month-tsibble---------------------------------------------------------
z %>%
  mutate(Month = yearmonth(Month)) %>%
  as_tsibble(index = Month)


## ----tstable2, echo=FALSE--------------------------------------------------
tribble(
  ~`Frequency`, ~Function,
  "Annual", "`start:end`",
  "Quarterly", "`yearquarter()`",
  "Monthly", "`yearmonth()`",
  "Weekly", "`yearweek()`",
  "Daily", "`as_date()`, `ymd()`",
  "Sub-daily", "`as_datetime()`"
) %>%
  knitr::kable(booktabs = TRUE)


## ----wide, include=FALSE---------------------------------------------------
options(width = 78)


## ----pbs1, dependson='wide'------------------------------------------------
gafa_stock


## ----pbs2------------------------------------------------------------------
gafa_stock %>%
  filter(Symbol == "FB")


## --------------------------------------------------------------------------
gafa_stock %>% select(Date,Close)


## ----pbs6------------------------------------------------------------------
gafa_stock_rtn <- gafa_stock %>%
  group_by_key() %>%
  select(Date,Close) %>% 
  mutate(simple_rtn = difference(Close)/lag(Close), 
         log_rtn = difference(log(Close))) %>% 
  filter(!is.na(log_rtn))


## ----a10, echo=FALSE, dependson="pbs6"-------------------------------------
gafa_stock_rtn


## ----narrow, include=FALSE-------------------------------------------------
options(width = 50)


## ----pbs4a-----------------------------------------------------------------
gafa_stock_rtn %>%
  group_by_key() %>%
  index_by(Month = ~yearmonth(.)) %>%
  summarise(mean_log_rtn = mean(log_rtn),
            multi_log_rtn = sum(log_rtn))


## --------------------------------------------------------------------------
gafa_stock_rtn %>%
    features(log_rtn, list(mean = mean, sd = sd, 
                           min = min, max = max))


## --------------------------------------------------------------------------
gafa_stock_rtn %>% as_tibble() %>%
  group_by(Symbol) %>%
    summarise(mean = mean(log_rtn), sd = sd(log_rtn),
              min = min(log_rtn), max = max(log_rtn))

