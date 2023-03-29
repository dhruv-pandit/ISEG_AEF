knitr::opts_chunk$set(
  echo = TRUE, message = FALSE, warning = FALSE, cache = FALSE,
  dev.args = list(pointsize = 11), fig.height=3.5, fig.width=7, fig.align='center'
)
options(digits = 3, width = 60)

rm(list=ls())

lapply(names(sessionInfo()$otherPkgs), function(pkgs)
  detach(
    paste0('package:', pkgs),
    character.only = T,
    unload = T,
    force = T
  ))


library(tidyverse)
library(fpp3)
library(tidyquant)
library(sandwich)
library(lmtest)
library(GGally)
library(kableExtra)


stock <- tidyquant::tq_get(c("^GSPC","^N225","^IXIC","^NYA"), 
   get = "stock.prices" ,from="1990-01-01", to="2012-12-31") %>% 
  mutate(symbol = ifelse(symbol == "^GSPC","S&P500",symbol)) %>% 
  mutate(symbol = ifelse(symbol == "^N225","Nikkei",symbol)) %>% 
  mutate(symbol = ifelse(symbol == "^IXIC","NASDAQ",symbol)) %>% 
  mutate(symbol = ifelse(symbol == "^NYA","NYSE",symbol)) %>% 
  mutate(date = as_date(date)) %>% 
  as_tsibble(index = date,key = symbol) %>% 
  group_by_key() %>% 
  mutate(rtn = difference(log(close))) %>% 
  filter(!is.na(rtn)) %>% 
  group_by_key() %>% 
  mutate(trading_day = row_number()) 

stock %>% select(symbol,date,close,rtn)

stock %>% 
  autoplot(close)+facet_grid(vars(symbol), scale = "free_y")+
  theme(legend.position="none", axis.text=element_text(size=6),
        axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

stock %>% 
  autoplot(rtn)+facet_grid(vars(symbol), scale = "free_y" )+
  theme(legend.position="none", axis.text=element_text(size=6),
        axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

stock %>%
  features(rtn,list(skew =  timeSeries::colSkewness, 
                    n = ~length(.))) %>%
             mutate(skew_stat =  sqrt(n)*skew/sqrt(6), 
                    p_val_skew_stat = 2*(1-pnorm(abs(skew_stat))))
             

stock %>%
  features(rtn,list(skew =  timeSeries::colSkewness, 
                    n = ~length(.))) %>% 
             mutate(skew_stat =  sqrt(n)*skew/sqrt(6), 
                    p_val_skew_stat = 2*(1-pnorm(abs(skew_stat)))) %>% 
  kbl(booktabs = TRUE)
             

stock %>%
  features(rtn,list(kurt =  timeSeries::colKurtosis, 
                    n = ~length(.))) %>% 
             mutate(kurt = kurt +3, 
                    kurt_stat =  sqrt(n)*kurt/sqrt(24), 
                    p_val_kurt_stat = 2*(1-pnorm(abs(kurt_stat))))
             

stock %>%
  features(rtn,list(kurt =  timeSeries::colKurtosis, 
                    n = ~length(.))) %>% 
             mutate(kurt = kurt+3,
                    kurt_stat =  sqrt(n)*kurt/sqrt(24), 
                    p_val_kurt_stat = 2*(1-pnorm(abs(kurt_stat)))) %>% 
  kbl(booktabs = TRUE)

feat_normal <- function(x){
  n <- length(x)
  kurt <- sum(((x-mean(x))/sd(x))^4)/n
  skew <- sum(((x-mean(x))/sd(x))^3)/n
  JB <- (n/6)*(skew^2) + (n/24)*((kurt-3)^2)
  pval_JB <- 1-pchisq(JB, df = 2)
  c(kurt = kurt, skew = skew, 
    JB = JB, pval_JB = pval_JB) 
}

register_feature(feat_normal, 
                 tags = c("simple", "summary"))

stock %>% features(rtn,feat_normal)

stock %>% features(rtn,feat_normal) %>% kbl(booktabs = TRUE)

stock %>%  group_by_key() %>% 
  index_by(Month = tsibble::yearmonth(date)) %>% 
  summarise(rtn = sum(rtn))%>% 
  features(rtn,feat_normal)

stock %>%  group_by_key() %>% 
  index_by(Month = tsibble::yearmonth(date)) %>% 
  summarise(rtn = sum(rtn))%>% 
  features(rtn,feat_normal) %>% kbl(booktabs = TRUE)

stock %>% group_by_key() %>% 
  index_by(Month = tsibble::yearquarter(date)) %>% 
  summarise(rtn = sum(rtn))%>% 
  features(rtn,feat_normal)

stock %>% 
  group_by_key() %>% 
  index_by(Month = tsibble::yearquarter(date)) %>% 
  summarise(rtn = sum(rtn))%>% 
  features(rtn,feat_normal) %>% kbl(booktabs = TRUE)

stock %>% 
  mutate(wkday = weekdays(date)) %>% 
  filter(symbol %in% c("Nikkei", "NASDAQ")) %>% 
  model(TSLM(rtn~wkday)) %>%  
  tidy() %>% select(-.model)

stock %>% 
  mutate(wkday = weekdays(date)) %>% 
  filter(symbol %in% c("Nikkei", "NASDAQ")) %>% 
  model(TSLM(rtn~wkday)) %>%  
  tidy() %>% select(-.model) %>% kbl(booktabs = TRUE)

stock %>% 
  mutate(wkday = weekdays(date)) %>% 
  filter(symbol == "S&P500") %>% 
  lm(rtn~1+wkday,data = .) %>% 
  lmtest::coeftest(vcov = sandwich::vcovHAC)

stock %>% 
  mutate(wkday = weekdays(date)) %>% 
  filter(symbol == "S&P500") %>% 
  lm(rtn~1+wkday,data = .) %>% 
  lmtest::coeftest(vcov = sandwich::vcovHAC) %>% 
  tidy() %>% 
  kbl(booktabs = TRUE)

stock %>% 
  mutate(wkday = weekdays(date)) %>% 
  filter(symbol == "S&P500") %>% 
  lm(rtn~1+wkday,data = .)-> lm1

stock %>% 
  mutate(wkday = weekdays(date)) %>% 
  filter(symbol == "S&P500") %>% 
  lm(rtn~1,data = .)-> lm2

  lmtest::waldtest(lm1, lm2, vcov = sandwich::vcovHC)

stock %>% update_tsibble(index = trading_day) %>% ACF(rtn) %>% 
  autoplot()+
  theme(legend.position="none", axis.text=element_text(size=6),
        axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

stock %>% features(rtn, 
                   list(~ljung_box(.,lag = 10),
                        ~box_pierce(.,lag = 10)
                        ))

stock %>% features(rtn, 
                   list(~ljung_box(.,lag = 10),
                        ~box_pierce(.,lag = 10)
                        ))%>% 
  kbl(booktabs = TRUE)

stock %>% features(rtn, 
                   list(unitroot_kpss,
                        unitroot_pp))

stock %>% features(rtn, 
                   list(unitroot_kpss,
                        unitroot_pp))%>% 
  kbl(booktabs = TRUE)

stock %>% 
  autoplot(rtn)+facet_grid(vars(symbol), scale = "free_y" )+
  theme(legend.position="none", axis.text=element_text(size=6),
        axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

knitr::include_graphics("sp500p.jpg")

stock %>% update_tsibble(index = trading_day) %>% ACF(rtn^2) %>% 
  autoplot()+
  ggtitle("")+
  theme(legend.position="none", axis.text=element_text(size=6),
        axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

stock %>% update_tsibble(index = trading_day) %>% ACF(abs(rtn)) %>% 
  autoplot()+
  theme(legend.position="none", axis.text=element_text(size=6),
        axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

stock %>% 
  as_tibble() %>% 
  group_by(symbol) %>% 
  summarise(corr = cor(rtn^2,lag(rtn),use='complete.obs'))

stock %>% 
  as_tibble() %>% 
  group_by(symbol) %>% 
  summarise(corr = cor(rtn^2,lag(rtn),use='complete.obs'))%>% 
  kbl(booktabs = TRUE)


stock %>% select(symbol,date,rtn) %>% 
  pivot_wider(values_from = rtn, names_from = symbol)%>% 
  GGally::ggpairs(2:5)

stock %>% mutate(rtn2 = rtn^2) %>% select(symbol,date,rtn2) %>% 
  pivot_wider(values_from = rtn2, names_from = symbol)%>% 
  GGally::ggpairs(2:5)
