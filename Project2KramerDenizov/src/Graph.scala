import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import Main.spark.implicits._ 
import plotly._, element._, layout._, Plotly._
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}

object Graph {
  
  
  def plotTweetsPerDay(teslaDS: Dataset[Tweet], savePlot: Boolean) = {
     
    val tweetsPerDayDS = teslaDS.groupBy("date")
	                              .agg(count("*") as "tweetsPerDay")
	                              .orderBy($"tweetsPerDay")
	                              
	  val tweetCount = tweetsPerDayDS.select('tweetsPerDay).collect.map(_.getAs[Long](0).toInt).toSeq;                            
	  val labels: Seq[String] = tweetsPerDayDS.select('date).collect.map(_.getAs[java.sql.Date](0).toString).toSeq;    
	            
	  val bar =  Bar(labels, tweetCount, name = "Tweets")
	  
	  if(savePlot){
  	  Seq(
       bar
      ).plot(
        title = "Tweets per Day - #tesla",
        path = "data/plots/plot_tesla_tweets_per_day.html"
      ) 
	  }
    
    bar
  }
  
  def plotStockPerDay(stockDS: Dataset[StockPrice], savePlot: Boolean): Bar = {
    val stockPrices = stockDS.select('close).collect.map(_.getAs[Double](0)).toSeq;                            
	  val labels: Seq[String] = stockDS.select('date).collect.map(_.getAs[java.sql.Date](0).toString).toSeq;    
	  
	  val bar = Bar(labels, stockPrices, name = "Stocks")
	  
	  if(savePlot){
	    Seq(
        bar
      ).plot(
        title = "TSLA Close Price per Day",
        path = "data/plots/plot_stock_price_per_day.html"
      )       
	  }
    
    bar
  }
  
  def plotStockAndTweetsPerDay(teslaDS: Dataset[Tweet], stockDS: Dataset[StockPrice]) = {
    
    //normalize tweet counts
    val tweetsPerDayDS = teslaDS.groupBy("date")
	                              .agg(count("*") as "tweetsPerDay")
	                              .orderBy($"tweetsPerDay")
    val tweetCountRDD: RDD[Double] = tweetsPerDayDS.select('tweetsPerDay).map(_.getAs[Long](0).toDouble).rdd
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(tweetCountRDD.map(x => Vectors.dense(x)))
    val normalizedCountsRDD: RDD[Vector] = tweetCountRDD.map(x => scaler.transform(Vectors.dense(x)))
    val normalizedCounts: Seq[Double] = normalizedCountsRDD.map(_.toArray(0)).collect.toSeq
    
    val tweetLabels: Seq[String] = tweetsPerDayDS.select('date).collect.map(_.getAs[java.sql.Date](0).toString).toSeq;    
  
    val tweetBar = Bar(tweetLabels, normalizedCounts, name = "Tweets")
    
    //normalize stock price
    
    val stockPricesRDD: RDD[Double] = stockDS.select('close).map(_.getAs[Double](0)).rdd;       
    val stockScaler = new StandardScaler(withMean = true, withStd = true).fit(stockPricesRDD.map(x => Vectors.dense(x)))
    val normalizedStockRDD: RDD[Vector] = stockPricesRDD.map(x => stockScaler.transform(Vectors.dense(x)))
    val normalizedStocks: Seq[Double] = normalizedStockRDD.map(_.toArray(0)).collect.toSeq
    
    val stockLabels: Seq[String] = stockDS.select('date).collect.map(_.getAs[java.sql.Date](0).toString).toSeq; 
    
    val stockBar = Bar(stockLabels, normalizedStocks, name = "Stocks")
    
    Seq(
      stockBar,
      tweetBar
    ).plot(
      title = "TSLA Close Price and Tweet Count per Day, Normalized",
      path = "data/plots/plot_stocks_and_tweets_day_normalized.html"
    )     
  }
}