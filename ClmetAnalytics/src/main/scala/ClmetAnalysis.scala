package clmet

import clmet.Common._
import java.io.File
import java.io.PrintWriter
import scala.io.Source
import scala.xml.XML

object ClmetAnalysis {

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
  
  def writeXmlsToDir(pathToDir: String, xmls: Vector[ClmetAnalysis]) = {
    xmls.map(xml => writeXmlToFile(pathToDir, xml))
  }
}

case class ClmetAnalysis(
    id: String,
    period: String,
    text: Vector[String]
)
