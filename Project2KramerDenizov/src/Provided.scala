import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import Main.spark.implicits._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.isnull

object Provided {
  
  def loadAirLineTweets(): Dataset[LabeledTweet] = {
    //CSV schema
    //tweet_id,airline_sentiment,airline_sentiment_confidence,negativereason,negativereason_confidence,
    //airline,airline_sentiment_gold,name,negativereason_gold,retweet_count,text,tweet_coord,tweet_created,tweet_location,user_timezone
  
    val convertSentiment: (String => Int) = (sentiment: String) => sentiment match {
      case "neutral" => 1
      case "positive" => 2
      case "negative" => 0
      case _ => 1
    }
    
    val sentimentFunc = udf(convertSentiment)
    
    //a filter for removing empty row
    def isNotEmptyUdf = udf[Boolean, String](text => text != null && !text.isEmpty)
    
    Main.spark.read
              .option("header", "true")
              .option("inferSchema", "true")
              .csv("data/provided/airline.csv")
              .withColumn("sentiment", sentimentFunc('airline_sentiment))
              .select('sentiment, 'text)
              .withColumnRenamed("sentiment", "label")
              .filter(isNotEmptyUdf('text))
              .as[LabeledTweet]
  }
}

case class LabeledTweet(label: Double, override val text: String) extends Tweet(text)