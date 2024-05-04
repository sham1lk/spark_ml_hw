package org.apache.spark.ml.made

import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.DataFrame


class LRTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta = 0.1
  lazy val data: DataFrame = LRTest._data


  "Model" should "work fine" in {

    val pipeline = new Pipeline().setStages(Array(
      new DistributedLinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
    ))

    val model = pipeline.fit(data)

    val predictions: Array[Double] = model.transform(data).collect().map(_.getAs[Double]("prediction"))
    implicit val encoder: Encoder[Double] = ExpressionEncoder()
    val labels = data.select(data("label").as[Double]).collect()

    predictions.length should be(labels.length)

    predictions(0) should be(labels(0) +- delta)
    predictions(1) should be(labels(1) +- delta)
  }
}

object LRTest extends WithSpark {

  lazy val points = Seq((Vectors.dense(1.0), 3.0), (Vectors.dense(2.0), 5.0))
  lazy val _data = {
    import sqlc.implicits._
    points.toDF("features", "label")
  }

  spark.sparkContext.setLogLevel("ERROR")
}