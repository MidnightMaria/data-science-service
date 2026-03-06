package com.agnesmaria.datascience.controller;

import com.agnesmaria.datascience.service.CSVReaderService;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping("/api/data-science")
public class DataScienceController {

    private final CSVReaderService csvReaderService;

    public DataScienceController(CSVReaderService csvReaderService) {
        this.csvReaderService = csvReaderService;
    }

    @GetMapping("/inventory-optimization")
    public List<Map<String,String>> inventoryOptimization() {

        return csvReaderService.readCSV(
            "reports/optimization/inventory_optimization_report.csv"
        );
    }

    @GetMapping("/demand-forecast")
    public List<Map<String,String>> demandForecast() {

        return csvReaderService.readCSV(
            "reports/forecast/future_demand_forecast.csv"
        );
    }

    @GetMapping("/dashboard")
    public Map<String,Integer> dashboard() {

        List<Map<String,String>> data =
            csvReaderService.readCSV(
                "reports/optimization/inventory_optimization_report.csv"
            );

        int orderNow = 0;
        int overstock = 0;
        int safe = 0;

        for (Map<String,String> row : data) {

            String status = row.get("status");

            if ("ORDER NOW".equals(status)) orderNow++;
            if ("OVERSTOCK".equals(status)) overstock++;
            if ("SAFE".equals(status)) safe++;
        }

        Map<String,Integer> result = new HashMap<>();

        result.put("items", data.size());
        result.put("orderNow", orderNow);
        result.put("overstock", overstock);
        result.put("safe", safe);

        return result;
    }

}