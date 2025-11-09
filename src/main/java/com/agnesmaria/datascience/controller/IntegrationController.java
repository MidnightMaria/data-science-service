package com.agnesmaria.datascience.controller;

import com.agnesmaria.datascience.service.DataFetchService;
import com.agnesmaria.datascience.service.DataExportService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/integration")
public class IntegrationController {

    private final DataFetchService dataFetchService;
    private final DataExportService dataExportService;


    // ✅ Manual constructor agar Spring bisa inject bean tanpa Lombok
    public IntegrationController(DataFetchService dataFetchService, DataExportService dataExportService) {
        this.dataFetchService = dataFetchService;
        this.dataExportService = dataExportService;
    }

    @GetMapping("/fetch-inventory")
    public String fetchInventory() {
        return dataFetchService.fetchInventoryData();
    }

    @GetMapping("/fetch-retail")
    public String fetchRetail() {
        return dataFetchService.fetchRetailData();
    }

    @GetMapping("/fetch-all")
    public String fetchAll() {
        return dataFetchService.fetchAll();
    }
    @GetMapping("/export-csv")
    public String exportCSV() {
        return dataExportService.exportToCSV();
    }

}
