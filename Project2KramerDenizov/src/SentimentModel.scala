import org.apache.spark.rdd.RDD
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp._
import java.util.Properties
import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer
import edu.stanford.nlp.ling.CoreAnnotations._
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations.SentimentAnnotatedTree
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations
import edu.stanford.nlp.ling._
import Main.spark.implicits._
import org.apache.spark.util.collection._
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.feature.Word2Vec


object SentimentModel {
  

  def computeSentiment(tweetDS: Dataset[LabeledTweet]): Dataset[SentimentTweet] = {
    println(s"Size is ${tweetDS.count}")
    tweetDS.mapPartitions{ part => 
	    val pipeline = createStandfordNLP
	    part.map(tweet => performSentiment(tweet, pipeline))
	  }
  }
  
  var count = 20
  
  def performSentiment(tweet : LabeledTweet, pipeline : StanfordCoreNLP) = {
    val emojiPattern = """[\p{So}+|\uFE0F]"""
    val handlePattern = """@[\w|_]+"""
    val text = tweet.text.replaceAll(emojiPattern, "").replaceAll(handlePattern, "")
    if(count > 0){
    println(text)
    count -= 1
    }
	  val document = new Annotation(text)
	  pipeline.annotate(document)
	  
	  val sentences = document.get(classOf[SentencesAnnotation])
	
	  val sentiments = for(sentence <- sentences; token <- sentence.get(classOf[TokensAnnotation])) yield {
	    val tree = sentence.get(classOf[SentimentAnnotatedTree]);
      val sentiment = RNNCoreAnnotations.getPredictedClass(tree) match {
        case 0 => -2 //very negative
        case 1 => -1 //negative
        case 2 => 0 //neutral
        case 3 => 1 //positive
        case 4 => 2 //very positive
      }
      sentiment
	  }
    
    val totalSentiment = sentiments.sum match {
      case negative if(negative < -3) => -1
      case positive if(positive > 1) => 1
      case neutral => 0
    }

	  SentimentTweet(tweet.label, tweet.text, totalSentiment.toDouble)
	}

  //creates and initializes the Stanford NLP object
	
	def createStandfordNLP() : StanfordCoreNLP = {
	  val stanfordProps = new Properties()
    stanfordProps.put("annotators", "tokenize, ssplit, parse, sentiment")
    new StanfordCoreNLP(stanfordProps)
	}
}

case class SentimentTweet(label: Double, text: String, sentiment: Double)
