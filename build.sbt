name := "lin_reg"

version := "0.1"

scalaVersion := "2.12.13"

val sparkVersion = "3.0.1"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.4.3",
  "org.apache.spark" %% "spark-mllib" % "3.4.3"
)

libraryDependencies += ("org.scalatest" %% "scalatest" % "3.2.2" % "test" withSources())