package org.apache.spark.ml.made

import org.apache.spark.ml.util._
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasPredictionCol}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}


trait LinearRegressionParams extends HasFeaturesCol with HasLabelCol with HasPredictionCol {
  def setFeaturesCol(value: String) : this.type = set(featuresCol, value)
  def setLabelCol(value: String) : this.type = set(labelCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  val iters = new IntParam(this, "iters", "Number of iterations")
  def getIters: Int = $(iters)
  def setIters(value: Int): this.type = set(iters, value)

  val learningRate = new DoubleParam(this, "learningRate", "Learning rate")
  def getLearningRate: Double = $(learningRate)
  def setLearningRate(value: Double): this.type = set(learningRate, value)

  setDefault(iters -> 1000, learningRate -> 1e-1)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getPredictionCol))
    }
  }
}


class DistributedLinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()
    implicit val encoder2 : Encoder[Double] = ExpressionEncoder()

    lazy val vectors: Dataset[(Vector, Double)] = dataset
      .select(dataset($(featuresCol)).as[Vector], dataset($(labelCol)).as[Double])

    val dim: Int = AttributeGroup.fromStructField(dataset.schema($(featuresCol))).numAttributes.getOrElse(
      vectors.first()._1.size + 1
    )
    val weights = breeze.linalg.DenseVector.rand[Double](dim)

    val iters: Int = getIters
    val learningRate: Double = getLearningRate
    val sampleCount: Double = vectors.count()
    val meanLearningRate: Double = learningRate / sampleCount

    var i: Int = 0
    while (iters > i) {
      val loss = vectors.rdd.mapPartitions(partition => {
        partition.map(row => {
          val x = new breeze.linalg.DenseVector[Double](1.0 +: row._1.toArray)
          val y = row._2
          ((x dot weights) - y) * x
        })
      }).reduce(_+_)

      weights -= meanLearningRate * loss
      i += 1
    }
    copyValues(new LinearRegressionModel(Vectors.fromBreeze(weights))).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[DistributedLinearRegression]

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val weights: DenseVector) extends Model[LinearRegressionModel]
  with LinearRegressionParams {

  def this(weights: Vector) =
    this(Identifiable.randomUID("linearRegressionModel"), weights.toDense)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(weights: DenseVector), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf =
      dataset.sqlContext.udf.register(uid + "_transform",
        (x: Vector) => {
          val vec = new DenseVector(1.0 +: x.toArray)
          vec.dot(weights)
        }
      )

    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}
