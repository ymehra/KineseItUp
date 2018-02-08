# run on 8/25/2015 #

rm(list=ls())
library(tree)
library(randomForest)
library(stringi)
library(data.table)
# change paths if necessary

###this is for ActiGraph files with 80Hz data- .csv files without timestamps. 
#directory for functions
directory <-"C:\\Users\\skeadle\\Desktop\\ACS\\"

#directory for ActiGraph files
ag.directory<-"S:/Kinesiology Keadle_Research/Activity Monitoring/Study Data/ActiGraph/wrt_raw_csv/"

#directory where processed files are written
out.directory<-'S:/Kinesiology Keadle_Research/Activity Monitoring/Study Data/analysis/ag/'

#name of on off log 
on.off<- paste(out.directory,"DO_LOG_final_ag_acsm.csv", sep="")


load("E:/hazda/john final/mods.RData")
source("E:/hazda/john final/fuctions.for.models.R")


start.stop <-  read.csv(on.off)
#temp <- Sys.glob(paste(directory,"sub",s,"_",v,"_",study,".csv",sep=""))

start.stop$date2 <-paste(start.stop$start_month,"/",start.stop$start_day,"/",start.stop$start_year,sep="")
start.stop$on.times <-  paste(start.stop$date2, start.stop$start_time)
start.stop$off.times <- paste(start.stop$date2, start.stop$stop_time)



## need to create on/off record with first time worn and last time off in a similar format to this. 
##The readfile only reads in data within that range. 

#First trial date on  First trial time on	First trial date off	first trial time off
#Subject First trial date on	First trial time on	First trial date off	first trial time off
#1001    18-Aug-14	          7:00 AM	           26-Aug-14	            6:54 AM
#1002    18-Aug-14	          7:00 AM	            26-Aug-14	            6:53 AM


n.sub <- dim(start.stop)[1]

win.width <- 15 # width of "windows" to analyze in seconds
filelist <- list.files(ag.directory)
iii<-1
for (iii in 1:length(filelist))
{
  
  
  sub <-   substr(filelist[iii],4,7)
  filename <- paste(ag.directory,filelist[iii],sep='')
  
  print(Sys.time())
  print(iii)
  
  sub.info <- subset(start.stop,start.stop$id==sub)
  
  
  start <- start.stop$on.times[1]
  start <- strptime(start,format= "%m/%d/%Y %H:%M:%S")
  
  stop <- start.stop$off.times[2]
  stop <- strptime(stop,format= "%m/%d/%Y %H:%M:%S")
  
  sub.info$on.times<-	strptime(sub.info$on.times,format="%m/%d/%Y %H:%M:%S")
  sub.info$off.times<-	strptime(sub.info$off.times, format="%m/%d/%Y %H:%M:%S")
  
  ag.data <- read.act.2(filename,start,stop) 
  n <- dim(ag.data)[1]
  
  head(ag.data)
  tail(ag.data)
  
  print(Sys.time())
  print(iii)
  
  
  mins <- ceiling(n/(80*win.width)) 
  ag.data$min <- rep(1:mins,each=win.width*80)[1:n]
  ag.data$vm <- sqrt(ag.data$V1^2+ag.data$V2^2+ag.data$V3^2)
  ag.data$v.ang <- 90*(asin(ag.data$V1/ag.data$vm)/(pi/2))
  ag.data.sum <- data.frame(mean.vm=tapply(ag.data$vm,ag.data$min,mean,na.rm=T),
                            sd.vm=tapply(ag.data$vm,ag.data$min,sd,na.rm=T),
                            mean.ang=tapply(ag.data$v.ang,ag.data$min,mean,na.rm=T),
                            sd.ang=tapply(ag.data$v.ang,ag.data$min,sd,na.rm=T),
                            p625=tapply(ag.data$vm,ag.data$min,pow.625),
                            dfreq=tapply(ag.data$vm,ag.data$min,dom.freq),
                            ratio.df=tapply(ag.data$vm,ag.data$min,frac.pow.dom.freq))
  
  ag.data.sum$start.time <- as.POSIXlt(tapply(ag.data$time,ag.data$min,min,na.rm=T),origin="1970-01-01 00:00.00 UTC")
  
  head(ag.data.sum)
  # applies models from JAP paper
  # ag.data.sum has estimaes over time
  ag.data.sum$METs.rf <- predict(rf.met.model,newdata=ag.data.sum)
  ag.data.sum$METs.lm <- predict(lm.met.model,newdata=ag.data.sum)
  ag.data.sum$METs.rf[ag.data.sum$sd.vm==0] <- 1
  ag.data.sum$METs.lm[ag.data.sum$sd.vm==0] <- 1
  
  ag.data.sum$MET.lev.rf <- predict(rf.met.level.model,newdata=ag.data.sum)
  ag.data.sum$MET.lev.tr <- predict(tr.met.level.model,newdata=ag.data.sum,type="class")
  ag.data.sum$sed.rf <- predict(rf.sed.model,newdata=ag.data.sum)
  ag.data.sum$sed.tr <- predict(tr.sed.model,newdata=ag.data.sum,type="class")
  ag.data.sum$loc.rf <- predict(rf.loc.model,newdata=ag.data.sum)
  ag.data.sum$loc.tr <- predict(tr.loc.model,newdata=ag.data.sum,type="class")
  ag.data.sum$combo.rf <- predict(rf.combo.model,newdata=ag.data.sum)
  ag.data.sum$combo.tr <- predict(tr.combo.model,newdata=ag.data.sum,type="class")
  
  
  head(ag.data.sum)
  
  # daily summaries
  ag.data.sum$day <- trunc(ag.data.sum$start.time,units="day")
  ag.data.sum$hour <- trunc(ag.data.sum$start.time,units="hour")
  
  head(ag.data.sum)
  
  
  n.junk <- dim(sub.info)[1]
  
  #if there is an on/off record for the file, go through and get rid of times that are not-wear
  if (n.junk>0)
  {start <- strptime(sub.info$on.times[1],format= "%Y-%m-%d %H:%M:%S")
  start
  
  end <- strptime(sub.info$off.times[n.junk], format="%Y-%m-%d %H:%M:%S")
  end
  
  # before first on time
  inds <- (1:n)[ag.data.sum$start.time<start]
  inds <- subset(inds, !is.na(inds))
  if (length(inds)>0)
  {ag.data.sum <- ag.data.sum[-inds,]}
  n <- dim(ag.data.sum)[1]
  
  # after last off time
  inds <- (1:n)[ag.data.sum$start.time>end]
  if (length(inds)>0)
  {ag.data.sum <- ag.data.sum[-inds,]}
  
  n <- dim(ag.data)[1]
  
  # other times
  for (jjj in 1:(dim(sub.info)[1]-1))
  {
    inds <- (1:n)[(ag.data.sum$start.time>sub.info$off.times[jjj])&(ag.data.sum$start.time<sub.info$on.times[(jjj+1)])]
    inds <- subset(inds, !is.na(inds))
    if (length(inds)>0)
    {ag.data.sum <- ag.data.sum[-inds,]}
    n <- dim(ag.data.sum)[1]
  }
  
  
  temp <- tapply(ag.data.sum$METs.rf,as.character(ag.data.sum$day),mean)
  head(temp)
  ag.data.daily.sum <- 
    data.frame(day=names(temp),
               
               min.per.day=tapply(ag.data.sum$METs.lm>0,as.character(ag.data.sum$day),sum)*15/60,
               
               mean.METs.rf=tapply(ag.data.sum$METs.rf,as.character(ag.data.sum$day),mean),
               mean.METs.lm=tapply(ag.data.sum$METs.lm,as.character(ag.data.sum$day),mean),
               
               L.min.rf=tapply(ag.data.sum$MET.lev.rf=="light",as.character(ag.data.sum$day),sum)*15/60,
               M.min.rf=tapply(ag.data.sum$MET.lev.rf=="moderate",as.character(ag.data.sum$day),sum)*15/60,
               V.min.rf=tapply(ag.data.sum$MET.lev.rf=="vigorous",as.character(ag.data.sum$day),sum)*15/60,
               
               L.min.tr=tapply(ag.data.sum$MET.lev.tr=="light",as.character(ag.data.sum$day),sum)*15/60,
               M.min.tr=tapply(ag.data.sum$MET.lev.tr=="moderate",as.character(ag.data.sum$day),sum)*15/60,
               V.min.tr=tapply(ag.data.sum$MET.lev.tr=="vigorous",as.character(ag.data.sum$day),sum)*15/60,
               
               sed.min.rf=tapply(ag.data.sum$sed.rf=="sedentary",as.character(ag.data.sum$day),sum)*15/60,
               non.sed.min.rf=tapply(ag.data.sum$sed.rf=="non-sedentary",as.character(ag.data.sum$day),sum)*15/60,
               
               sed.min.tr=tapply(ag.data.sum$sed.tr=="sedentary",as.character(ag.data.sum$day),sum)*15/60,
               non.sed.min.tr=tapply(ag.data.sum$sed.tr=="non-sedentary",as.character(ag.data.sum$day),sum)*15/60,
               
               loc.min.rf=tapply(ag.data.sum$loc.rf=="locomotion",as.character(ag.data.sum$day),sum)*15/60,
               non.loc.min.rf=tapply(ag.data.sum$loc.rf=="non-locomotion",as.character(ag.data.sum$day),sum)*15/60,
               
               loc.min.tr=tapply(ag.data.sum$loc.tr=="locomotion",as.character(ag.data.sum$day),sum)*15/60,
               non.loc.min.tr=tapply(ag.data.sum$loc.tr=="non-locomotion",as.character(ag.data.sum$day),sum)*15/60)
  
  
  ag.data.sum$session <- 1 
  ag.data.sum$session[ag.data.sum$start.time>	sub.info$on.times[2]] <- 2
  summary(ag.data.sum$session)
  
  
  temp <- tapply(ag.data.sum$METs.rf,as.character(ag.data.sum$session),mean)
  head(temp)
  ag.data.session.sum <- 
    data.frame(session=names(temp),
               
               min.per.day=tapply(ag.data.sum$METs.lm>0,as.character(ag.data.sum$session),sum)*15/60,
               
               mean.METs.rf=tapply(ag.data.sum$METs.rf,as.character(ag.data.sum$session),mean),
               mean.METs.lm=tapply(ag.data.sum$METs.lm,as.character(ag.data.sum$session),mean),
               
               L.min.rf=tapply(ag.data.sum$MET.lev.rf=="light",as.character(ag.data.sum$session),sum)*15/60,
               M.min.rf=tapply(ag.data.sum$MET.lev.rf=="moderate",as.character(ag.data.sum$session),sum)*15/60,
               V.min.rf=tapply(ag.data.sum$MET.lev.rf=="vigorous",as.character(ag.data.sum$session),sum)*15/60,
               
               L.min.tr=tapply(ag.data.sum$MET.lev.tr=="light",as.character(ag.data.sum$session),sum)*15/60,
               M.min.tr=tapply(ag.data.sum$MET.lev.tr=="moderate",as.character(ag.data.sum$session),sum)*15/60,
               V.min.tr=tapply(ag.data.sum$MET.lev.tr=="vigorous",as.character(ag.data.sum$session),sum)*15/60,
               
               sed.min.rf=tapply(ag.data.sum$sed.rf=="sedentary",as.character(ag.data.sum$session),sum)*15/60,
               non.sed.min.rf=tapply(ag.data.sum$sed.rf=="non-sedentary",as.character(ag.data.sum$session),sum)*15/60,
               
               sed.min.tr=tapply(ag.data.sum$sed.tr=="sedentary",as.character(ag.data.sum$session),sum)*15/60,
               non.sed.min.tr=tapply(ag.data.sum$sed.tr=="non-sedentary",as.character(ag.data.sum$session),sum)*15/60,
               
               loc.min.rf=tapply(ag.data.sum$loc.rf=="locomotion",as.character(ag.data.sum$session),sum)*15/60,
               non.loc.min.rf=tapply(ag.data.sum$loc.rf=="non-locomotion",as.character(ag.data.sum$session),sum)*15/60,
               
               loc.min.tr=tapply(ag.data.sum$loc.tr=="locomotion",as.character(ag.data.sum$session),sum)*15/60,
               non.loc.min.tr=tapply(ag.data.sum$loc.tr=="non-locomotion",as.character(ag.data.sum$session),sum)*15/60)
  
  ag.data.session.sum$id <-sub
  
  #junk <- ag.data.daily.sum[ag.data.daily.sum$min.per.day>1000,]
  
  
  
  if (iii==1)
  {write.table(ag.data.session.sum, file=paste(out.directory,"do_agoutput_wrist_new.csv", sep=""), row.names=F, append=F, sep=",")}
  
  if (iii>1)
  {write.table(ag.data.session.sum, file=paste(out.directory,"do_agoutput_wrist_new.csv", sep=""), row.names=F, append=T, col.names=F, sep=",")}
  
  }	
  print(Sys.time())
  print(iii)
  
}



