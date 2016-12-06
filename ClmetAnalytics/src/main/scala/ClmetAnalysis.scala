package clmet

import clmet.Common._
import java.io.File
import java.io.PrintWriter
import scala.io.Source
import scala.xml.XML

trait OutputAnalysis{
  def writeXmlToFile(outputPath: String, xml: ClmetAnalysis): Unit
}

object PeriodOutputAnalysis extends OutputAnalysis {
  def writeXmlToFile(outputPath: String, xml: ClmetAnalysis) = {
    val file = new File(outputPath + s"/${xml.period}/${xml.id}.txt")
    val printWriter = new PrintWriter(file)
    for {
      paragraph <- xml.text
    } yield {
      printWriter.println(paragraph)
    }
    printWriter.close()
  }
}

object DecadeOutputAnalysis extends OutputAnalysis {
  val SpecificPeriods = Vector((1710,1740),(1740,1770),(1770,1800),(1800,1830),(1830,1860),(1860,1890),(1890,1920))
  
  def periodFromDecade(decade: Int) = {
    def isBetweenPeriod(periods: (Int,Int), decade: Int): Boolean = {
      val start = periods._1
      val end = periods._2
      start >= decade && decade < end
    }
    SpecificPeriods.find(periods => isBetweenPeriod(periods,decade))
  }
  
  def writeXmlToFile(outputPath: String, xml: ClmetAnalysis) = {
    for {
      decade <- xml.decade
      (start, end) <- periodFromDecade(decade)
    } yield {
      val file = new File(outputPath + s"/${start}-${end}/${xml.id}.txt")
      val printWriter = new PrintWriter(file)
      for {
        paragraph <- xml.text
      } yield {
        printWriter.println(paragraph)
      }
      printWriter.close()
    }
  }
}

case class ClmetAnalyzer(outputAnalysisType: OutputAnalysis) {

  def parseFiles(files: List[File]): Vector[ClmetAnalysis] = {
    files.flatMap(file => parseXmlFromFile(file)).toVector
  }

  def getAllFilesInDirectory(pathToDir: String): List[File] = {
    val d = new File(pathToDir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }
  
  def normalizeXml(xml: Vector[String]) = {
    val regex = "<page[ 0-9A-Za-z\"=]*/>|[&\";]".r
    val s = "<header>\n" + xml.reduceOption((a,b) => s"$a\n$b").getOrElse("") + "\n</header>"
    regex.replaceAllIn(s, "")
  }
  
  def parseXmlFromFile(file: File): Vector[ClmetAnalysis] = {
    val lines =  Source.fromFile(file).getLines.toVector
    val normalizedLines = normalizeXml(lines)
    val xml = XML.loadString(normalizedLines)
    ClmetParser.parseXml(xml)
  }
  
  def parseAllXmlFromDir(pathToDir: String): Vector[ClmetAnalysis] = {
    val files = getAllFilesInDirectory(pathToDir)
    parseFiles(files)
  }
  
  def writeXmlsToDir(pathToDir: String, xmls: Vector[ClmetAnalysis]) = {
    xmls.map(xml => outputAnalysisType.writeXmlToFile(pathToDir, xml))
  }
}

case class ClmetAnalysis(
    id: String,
    year: Option[Int] = None,
    decade: Option[Int] = None,
    period: String,
    text: Vector[String]
)
