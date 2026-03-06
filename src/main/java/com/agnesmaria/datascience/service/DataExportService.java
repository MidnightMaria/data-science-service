package com.agnesmaria.datascience.service;

import com.agnesmaria.datascience.config.IntegrationConfig;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.opencsv.CSVWriter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

@Service
public class DataExportService {

    private static final Logger log = LoggerFactory.getLogger(DataExportService.class);
    private final RestTemplate restTemplate;
    private final IntegrationConfig integrationConfig;
    private final ObjectMapper mapper = new ObjectMapper();

    public DataExportService(RestTemplate restTemplate, IntegrationConfig integrationConfig) {
        this.restTemplate = restTemplate;
        this.integrationConfig = integrationConfig;
    }

    public String exportToCSV() {
        log.info("🚀 Fetching and exporting data from Inventory and Retail services...");

        try {
            // Fetch data dari kedua service
            String inventoryJson = restTemplate.getForObject(
                    integrationConfig.getInventoryService().getUrl(), String.class);
            String retailJson = restTemplate.getForObject(
                    integrationConfig.getRetailService().getUrl(), String.class);

            // Convert JSON string → List of Map
            List<Map<String, Object>> inventoryData =
                    mapper.readValue(inventoryJson, new TypeReference<>() {});
            List<Map<String, Object>> retailData =
                    mapper.readValue(retailJson, new TypeReference<>() {});

            // Simpan ke CSV
            String inventoryPath = "data/processed/inventory_data.csv";
            String retailPath = "data/processed/retail_data.csv";

            saveAsCSV(inventoryData, inventoryPath);
            saveAsCSV(retailData, retailPath);

            return String.format("""
                    Export success!
                    Inventory CSV: %s
                    Retail CSV: %s
                    """, inventoryPath, retailPath);

        } catch (Exception e) {
            log.error("Error exporting data to CSV", e);
            return "Failed to export data: " + e.getMessage();
        }
    }

    private void saveAsCSV(List<Map<String, Object>> data, String path) throws IOException {
        if (data.isEmpty()) {
            log.warn("No data to save for {}", path);
            return;
        }

        Files.createDirectories(Paths.get("data/processed"));

        try (CSVWriter writer = new CSVWriter(new FileWriter(path))) {
            // Header
            Set<String> headers = data.get(0).keySet();
            writer.writeNext(headers.toArray(new String[0]));

            // Rows
            for (Map<String, Object> row : data) {
                String[] values = headers.stream()
                        .map(key -> Objects.toString(row.get(key), ""))
                        .toArray(String[]::new);
                writer.writeNext(values);
            }
        }
        log.info("CSV saved to {}", path);
    }
}
