df=read.csv("C://Users/User/Documents/kppdata.csv")
df=df[c('ds','y')]
append(df)
plot(y~ds)
par(mfrow=c(1,1))

#which period? [event-before[i]]~[event-(before[i]+length)]

#beforeIndex=0 : 1day before the event
beforeIndex=(0:19)

#length=7,14,21...
length=7


#get the index of one day before the event

#chusuk: dates that are one day before chusuk(#is.element(x, y) is identical to x %in% y)
beforeEvent<-which(df$ds %in% c('2010-09-21 0:00','2011-09-11 0:00','2012-09-29 0:00','2013-09-18 0:00','2014-09-07 0:00','2015-09-26 0:00','2016-09-13 0:00', '2017-10-01 0:00'))

#pp
#beforeEvent<-which(df$ds %in% c('2010-11-10 0:00','2011-11-10 0:00','2012-11-11 0:00','2013-11-10 0:00','2014-11-10 0:00','2015-11-10 0:00','2016-11-10 0:00', '2017-11-10 0:00'))

#christmas
#beforeEvent<-which(df$ds %in% c('2010-12-24 0:00','2011-12-24 0:00','2012-12-24 0:00','2013-12-24 0:00','2014-12-24 0:00','2015-12-24 0:00','2016-12-24 0:00'))

#newyear
beforeEvent<-which(df$ds %in% c('2011-02-02 0:00','2012-01-22 0:00','2013-02-09 0:00','2014-01-30 0:00','2015-02-18 0:00','2016-02-07 0:00', '2017-01-27 0:00'))


tTest<-function(a){
  eventIndex=c()
  for (i in (1:length(beforeEvent))){eventIndex=c(eventIndex,(beforeEvent[i]-(a+length)):(beforeEvent[i]-a))}
  
  event<-df[eventIndex,]
  #나중에 여유되면 시도해보겠음 ㅜ notin이 되면 좋을텐ㄷ
  #notEvent<-subset(df,index %notin% eventIndex)
  event$y
  #mean of Event
  eventMean<-mean(df[eventIndex,]$y)
  
  #mean of notEvent
  eventSum<-sum(df[eventIndex,]$y)
  totalSum<-sum(df$y)
  totalSum
  notEvent<-totalSum - eventSum
  notEventMean<-notEvent / (length(df[,1])-length(eventIndex))
  
  eventMean
  notEventMean
  if(eventMean>notEventMean){
    answer=t.test(event$y, mu=notEventMean, alternative="greater")$p.value
  }else answer=t.test(event$y, mu=notEventMean, alternative="less")$p.value
  return(answer)
}

k=c()
for (i in beforeIndex){
  print(tTest(i))
  #append(k,tTest(i))
  k<-c(k,tTest(i))
}

plot(k~c(beforeIndex))

plot(k~c(beforeIndex), type='l')


















a=4
eventIndex=c()
for (i in (1:length(beforeEvent))){eventIndex=c(eventIndex,(beforeEvent[i]-(a+13)):(beforeEvent[i]-a))}

event<-df[eventIndex,]
#나중에 여유되면 시도해보겠음 ㅜ notin이 되면 좋을
#notEvent<-subset(df,index %notin% eventIndex)
event$y
#mean of Event
eventMean<-mean(df[eventIndex,]$y)

#mean of notEvent
eventSum<-sum(df[eventIndex,]$y)
totalSum<-sum(df$y)
totalSum
notEvent<-totalSum - eventSum
notEventMean<-notEvent / (length(df[,1])-length(eventIndex))

eventMean
notEventMean
if(eventMean>notEventMean){
  t.test(event$y, mu=notEventMean, alternative="greater")
}else t.test(event$y, mu=notEventMean, alternative="less")


par(mfrow=c(1,2))
df_asDate<-df
df_asDate$ds=as.Date(df_asDate$ds)
graph16<-subset(df_asDate, df_asDate$ds<='2016-12-31' & df_asDate$ds>='2016-01-01')
plot(graph16$y~graph16$ds,type='l')
graph15<-subset(df_asDate, df_asDate$ds<='2015-12-31' & df_asDate$ds>='2015-01-01')
plot(graph15$y~graph15$ds,type='l')
graph14<-subset(df_asDate, df_asDate$ds<='2014-12-31' & df_asDate$ds>='2014-01-01')
plot(graph14$y~graph14$ds,type='l')
graph13<-subset(df_asDate, df_asDate$ds<='2013-12-31' & df_asDate$ds>='2013-01-01')
plot(graph13$y~graph13$ds,type='l')
graph12<-subset(df_asDate, df_asDate$ds<='2012-12-31' & df_asDate$ds>='2012-01-01')
plot(graph12$y~graph12$ds,type='l')
graph11<-subset(df_asDate, df_asDate$ds<='2011-12-31' & df_asDate$ds>='2011-01-01')
plot(graph11$y~graph11$ds,type='l')
graph10<-subset(df_asDate, df_asDate$ds<='2010-12-31' & df_asDate$ds>='2010-01-01')
plot(graph10$y~graph10$ds,type='l')


