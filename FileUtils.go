package main

import (
  "strconv"
  "strings"
	"io/ioutil"
)

func getData(file string) [][]float64 {

  allData := make([][]float64, 0)
  r, _ := ioutil.ReadFile(file)

  re := string(r)
  record := strings.Split(re, "\n")


  for i := 0; i < len(record); i++ {
    dataLine := make([]float64, 0)
    dataPoint := strings.Split(record[i], ",")
    for j := 0; j < len(dataPoint); j++ {
      val, _ := strconv.ParseFloat(dataPoint[j], 64)
      if val != 0 {
        dataLine = append(dataLine, val)
      }
    }
    if i < len(record) - 1 {
      startIndex := strings.LastIndexByte(record[i], ',')
      classString := record[i][startIndex + 1 : startIndex+ 2]
      classString = strings.Trim(classString, " ")
      classFloat, _ := strconv.ParseFloat(classString, 64)
      if classFloat == 2 {
        classFloat = 0
      }
      dataLine = append(dataLine, classFloat)
    }
    if len(dataLine) != 0 {
      allData = append(allData, dataLine)
    }
  }
  return allData
}


func prepareData(data [][]float64) [][]float64 {

  maxValues := make([]float64, len(data[0]) - 1)

  for i := 0; i < len(data); i++ {
    for j := 0; j < len(data[i]) - 1; j++ {
      if data[i][j] > maxValues[j] {
        maxValues[j] = data[i][j]
      }
    }
  }
  for i := 0; i < len(data); i++ {
    for j := 0; j < len(data[i]) - 1; j++ {
      data[i][j] = data[i][j] / maxValues[j]
    }
  }
  return data
}
