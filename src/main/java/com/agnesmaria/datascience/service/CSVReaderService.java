package com.agnesmaria.datascience.service;

import com.opencsv.CSVReader;
import org.springframework.stereotype.Service;

import java.io.FileReader;
import java.util.*;

@Service
public class CSVReaderService {

    public List<Map<String, String>> readCSV(String path) {

        List<Map<String, String>> data = new ArrayList<>();

        try (CSVReader reader = new CSVReader(new FileReader(path))) {

            String[] headers = reader.readNext();
            String[] line;

            while ((line = reader.readNext()) != null) {

                Map<String, String> row = new HashMap<>();

                for (int i = 0; i < headers.length; i++) {
                    row.put(headers[i], line[i]);
                }

                data.add(row);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        return data;
    }
}