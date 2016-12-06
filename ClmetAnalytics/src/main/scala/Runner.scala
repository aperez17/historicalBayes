package clmet

import ClmetAnalysis._

object Runner extends App {
  def processArg(arg: String): String = {
    arg.toLowerCase.trim.replaceAll("-+","")
  }
  def normalizeArg(arg: String): String = {
    arg.replaceAll(" +", " ")
  }

  def groupCommands(args: String) = {
    args.split(" -").foldLeft(Map.empty[String, String]) {
      case (map, arg) =>
        normalizeArg(arg).split(" ").map(_.trim) match {
          case Array(cmd, value) => map + (processArg(cmd) -> value)
          case Array(cmd) if cmd.nonEmpty => map + (processArg(cmd) -> "")
          case _ => map
        }
    }
  }

  val argMap = groupCommands(args.reduceOption((a, b) => s"$a $b").getOrElse(""))

  val inputFile = argMap.get("input").orElse(argMap.get("i"))
  val outputFile = argMap.get("output").orElse(argMap.get("o")).getOrElse("output.txt")
  val oneAtATime = argMap.get("s")
  val timePeriod = argMap.get("t")

  inputFile.map { case path =>
    if (oneAtATime.isEmpty) {
      val analyzer = ClmetAnalyzer(PeriodOutputAnalysis)
      val parsedXmls = analyzer.parseAllXmlFromDir(path)
      analyzer.writeXmlsToDir(outputFile, parsedXmls)
    } else {
      if (timePeriod.nonEmpty) {
        val analyzer = ClmetAnalyzer(DecadeOutputAnalysis)
        val file = new java.io.File(path)
        val parsedXml = analyzer.parseXmlFromFile(file)
        analyzer.writeXmlsToDir(outputFile, parsedXml)
      } else {
        val analyzer = ClmetAnalyzer(PeriodOutputAnalysis)
        val file = new java.io.File(path)
        val parsedXml = analyzer.parseXmlFromFile(file)
        analyzer.writeXmlsToDir(outputFile, parsedXml)
      }
    }
  }
}