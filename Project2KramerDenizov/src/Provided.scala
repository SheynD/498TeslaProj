import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import Main.spark.implicits._
import org.apache.spark.sql.functions.udf
 
object Provided {
  
  def loadAirLineTweets(): Dataset[SentimentTweet] = {
    //CSV schema
    //tweet_id,airline_sentiment,airline_sentiment_confidence,negativereason,negativereason_confidence,
    //airline,airline_sentiment_gold,name,negativereason_gold,retweet_count,text,tweet_coord,tweet_created,tweet_location,user_timezone
  
    val convertSentiment: (String => Int) = (sentiment: String) => sentiment match {
      case "neutral" => 0
      case "positive" => 1
      case "negative" => -1
    }
    
    val sentimentFunc = udf(convertSentiment)
  
    Main.spark.read
              .option("header", "true")
              .option("inferSchema", "true")
              .csv("data/provided/airline.csv")
              .withColumn("sentiment", sentimentFunc('airline_sentiment))
              .select('sentiment, 'text)
              .as[SentimentTweet]
  }
}

case class SentimentTweet(sentiment: Int, text: String)