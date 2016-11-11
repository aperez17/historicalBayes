val xmlParser = "org.scala-lang.modules" % "scala-xml_2.11" % "1.0.5"
mainClass in Compile := Some("clmet.Runner")
lazy val root = (project in file(".")).
  settings(
    name := "ClmetAnalytics",
    version := "1.0",
    scalaVersion := "2.11.8",
    libraryDependencies += xmlParser,
    mainClass in assembly := Some("clmet.Runner")
    )
