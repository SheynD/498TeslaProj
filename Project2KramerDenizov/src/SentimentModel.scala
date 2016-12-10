import org.apache.spark.rdd.RDD
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp._
import java.util.Properties
import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer
import edu.stanford.nlp.ling.CoreAnnotations._
import edu.stanford.nlp.ling._
import Main.spark.implicits._
import org.apache.spark.util.collection._
import org.apache.spark.sql.Dataset

object SentimentModel {
  

  def computeSentiment(tweetDS: Dataset[SentimentTweet]) = {
    tweetDS.m
  }
  
  
  
  
  
  
  
  
  //creates and initializes the Stanford NLP object
	def createStandfordNLP() : StanfordCoreNLP = {
	  val stanfordProps = new Properties()
    stanfordProps.put("annotators", "tokenize, ssplit, parse, sentiment")
    new StanfordCoreNLP(stanfordProps)
	}
}