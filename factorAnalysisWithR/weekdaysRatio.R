df=read.csv("C://Users/User/Documents/kppdata.csv")
df=df[c('ds','y')]

ratio<-c()
for (i in 0:6){
  a<-subset(df,strftime(df$ds,'%A')=="월요일")
  ratio<-append(ratio,  mean(a$y))
}


#friday: 0


a<-subset(df,strftime(df$ds,'%A')=="월요일")
print(mean(a$y))
a<-subset(df,strftime(df$ds,'%A')=="화요일")
print(mean(a$y))
a<-subset(df,strftime(df$ds,'%A')=="수요일")
print(mean(a$y))
a<-subset(df,strftime(df$ds,'%A')=="목요일")
print(mean(a$y))
a<-subset(df,strftime(df$ds,'%A')=="금요일")
print(mean(a$y))
a<-subset(df,strftime(df$ds,'%A')=="토요일")
print(mean(a$y))
a<-subset(df,strftime(df$ds,'%A')=="일요일")
print(mean(a$y))



ratio<-c(86808.36,98024.09,97411.06,95868.5,99652.41,56088.43,457.4715)
mean(ratio)
ratio/mean(ratio)