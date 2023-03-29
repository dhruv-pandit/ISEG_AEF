
 library(tidyverse)
 library(fpp3)
 library(kableExtra)


 stock <- tidyquant::tq_get(c("^GSPC","^IXIC","^NYA"),
    get = "stock.prices" ,from="2006-01-01", to="2012-12-31") %>%
   mutate(symbol = ifelse(symbol == "^GSPC","SP500",symbol)) %>%
   mutate(symbol = ifelse(symbol == "^IXIC","NASDAQ",symbol)) %>%
   mutate(symbol = ifelse(symbol == "^NYA","NYSE",symbol)) %>%
   mutate(date = as_date(date)) %>%
   as_tsibble(index = date,key = symbol) %>%
   group_by_key() %>%
   mutate(rtn = difference(log(close))) %>%
   group_by_key() %>%
   mutate(trading_day = row_number()) %>%
   select(symbol,date,rtn) %>%
   filter(!is.na(rtn)) %>%
   pivot_wider(names_from = "symbol",values_from = "rtn")

 stock

 
 library(MTS)
 
 stock[,2:4] %>% MTS::MarchTest()


 
 stock[,2:4] %>%
   mutate(NASDAQ=NASDAQ-mean(NASDAQ),
          NYSE=NYSE-mean(NYSE),
          SP500=SP500-mean(SP500)) %>%
   MTS::MarchTest()


 
 resid_VAR <- MTS::VAR(stock[,2:4], p = 1, output = F)$residuals
 MTS::MarchTest(resid_VAR)


   m1 <- MTS::VAR(stock[,2:4], p=1, output = F)
   m2 <- MTS::EWMAvol(m1$residuals)
   m2 <- MTS::EWMAvol(m1$residuals, lambda = -0.1)
 
 
 MTS::MCHdiag(m1$residuals,m2$Sigma.t)


   colnames(m2$Sigma.t) <- c("sigma11","sigma21","sigma31",
                              "sigma12","sigma22","sigma32",
                              "sigma13","sigma23","sigma33")
 
   tibble(date = stock$date[-1],as_tibble(m2$Sigma.t)) %>%
     as_tsibble(index = date) %>%
     pivot_longer(-date) %>%
     autoplot(value)+
     facet_wrap(~name,nrow = 3, ncol = 3)+
     theme_classic()+
     theme(legend.position = "none")

 colnames(m2$Sigma.t) <- c("sigma11","sigma21","sigma31",
                              "sigma12","sigma22","sigma32",
                              "sigma13","sigma23","sigma33")
 
   tibble(date = stock$date[-1],as_tibble(m2$Sigma.t)) %>%
     as_tsibble(index = date) %>%
     pivot_longer(-date) %>%
     autoplot(value)+
     facet_wrap(~name,nrow = 3, ncol = 3)+
     theme_classic()+
     theme(legend.position = "none")

 spec <-  rugarch::ugarchspec(variance.model=list(model="sGARCH",
                             garchOrder=c(1,1)),
                           mean.model=list(armaOrder=c(0,0),
                                           include.mean=FALSE))
 
 m11 <- rugarch::ugarchfit(data = stock$NASDAQ, spec = spec)
 m22 <- rugarch::ugarchfit(data = stock$NYSE, spec = spec)
 m33 <- rugarch::ugarchfit(data = stock$SP500, spec = spec)
 m12 <- rugarch::ugarchfit(data = stock$NASDAQ*stock$NYSE,
                           spec = spec)
 m13 <- rugarch::ugarchfit(data = stock$NASDAQ*stock$SP500,
                           spec = spec)
 m23 <- rugarch::ugarchfit(data = stock$NYSE*stock$SP500,
                           spec = spec)


 m1 <- MTS::VAR(stock[,2:4], p=1)
 m2 <- MTS::BEKK11(m1$residuals)

 MTS::MCHdiag(m1$residuals,m2$Sigma.t)

  colnames(m1$Sigma.t) <- c("sigma11","sigma21","sigma31",
                              "sigma12","sigma22","sigma32",
                              "sigma13","sigma23","sigma33")
 
 
   tibble(date = stock$date[-1],as_tibble(m1$Sigma.t)) %>%
     as_tsibble(index = date) %>%
     pivot_longer(-date) %>%
     autoplot(value)+
     facet_wrap(~name,nrow = 3, ncol = 3)+
     theme_classic()+
     theme(legend.position = "none")

  colnames(m1$Sigma.t) <- c("sigma11","sigma21","sigma31",
                              "sigma12","sigma22","sigma32",
                              "sigma13","sigma23","sigma33")
 
   tibble(date = stock$date[-1],as_tibble(m1$Sigma.t)) %>%
     as_tsibble(index = date) %>%
     pivot_longer(-date) %>%
     autoplot(value)+
     facet_wrap(~name,nrow = 3, ncol = 3)+
     theme_classic()+
     theme(legend.position = "none")

