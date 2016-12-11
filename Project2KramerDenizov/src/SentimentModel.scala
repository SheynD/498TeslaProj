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
import org.apache.spark.ml.linalg.Vector

object SentimentModel {
  val emojiPattern = """[\p{So}+|\uFE0F]"""
  val handlePattern = """@[\w|_]+"""

  def computeSentiment(tweetDS: Dataset[LabeledTweet]) = {
    tweetDS.mapPartitions{ part => 
	    val pipeline = createStandfordNLP
	    part.map(tweet => performSentiment(tweet, pipeline))
	  }
  }
  

  def performSentiment(tweet : Tweet, pipeline : StanfordCoreNLP) = {
  
    val text = tweet.text.replaceAll(emojiPattern, "")
                         .replaceAll(handlePattern, "")

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
      case negative if(negative < -3) => 0
      case positive if(positive > 1) => 2
      case neutral => 1
    }

    tweet match{
      case labeledTweet : LabeledTweet => SentimentTweet(tweet.asInstanceOf[LabeledTweet].label, tweet.text, totalSentiment.toDouble)
    }
	}
  //creates and initializes the Stanford NLP object
	def createStandfordNLP() : StanfordCoreNLP = {
	  val stanfordProps = new Properties()
    stanfordProps.put("annotators", "tokenize, ssplit, parse, sentiment")
    new StanfordCoreNLP(stanfordProps)
	}
	
	def computeWord2Vec(tweetDS: Dataset[LabeledTweet]) = {
	  val word2vec = new Word2Vec()
  	                    .setInputCol("wordsArray")
  	                    .setOutputCol("result")
                        .setVectorSize(50)
                        .setMinCount(0)
                        
    val tweetDSWithArray: Dataset[TempWord2VecTweet] = tweetDS.map{ labeledTweet => 
      val wordsArray = labeledTweet.text.replaceAll(handlePattern, "").split(" ").filterNot(_.isEmpty)
      TempWord2VecTweet(labeledTweet.label, labeledTweet.text, wordsArray)
    }

	  val model = word2vec.fit(tweetDSWithArray)
	  model.transform(tweetDSWithArray)
	}
	
	
}

class Tweet(val text: String)
case class SentimentTweet(label: Double, override val text: String, sentiment: Double) extends Tweet(text)
case class LabeledWord2VecTweet(label: Double, override val text: String, vector: Vector) extends Tweet(text)
case class TempWord2VecTweet(label: Double, text: String, wordsArray: Array[String])                   
