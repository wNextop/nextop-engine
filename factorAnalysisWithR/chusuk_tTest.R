df=read.csv("C://Users/User/Documents/kppdata.csv")
df=df[c('ds','y')]
append(df)
plot(y~ds)

#get the index of one day before the event

#chusuk (#is.element(x, y) is identical to x %in% y)
beforeEvent<-which(df$ds %in% c('2010-09-21 0:00','2011-09-11 0:00','2012-09-29 0:00','2013-09-18 0:00','2014-09-07 0:00','2015-09-26 0:00','2016-09-13 0:00', '2017-10-01 0:00'))

#pp
beforeEvent<-which(df$ds %in% c('2010-11-11 0:00','2011-11-11 0:00','2012-11-11 0:00','2013-11-11 0:00','2014-11-11 0:00','2015-11-11 0:00','2016-11-11 0:00', '2017-11-11 0:00'))

a=3

#ㅂㄷㅂㄷ 이게 왜 (1:7)은 안되고 (1:8)해야 되는지 알수없음 ㅜㅜ 알려줘
eventIndex=c()
#for (i in (1:7)){eventIndex=c(eventIndex,(beforeEvent[i]-6):beforeEvent[i])}
for (i in (1:8)){eventIndex=c(eventIndex,(beforeEvent[i]-(a+6)):(beforeEvent[i]-a))}

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
  t.test(event$y, mu=notEventMean, alternative="greater")
}
#else
#  t.test(event$y, mu=notEventMean, alternative="less")

  