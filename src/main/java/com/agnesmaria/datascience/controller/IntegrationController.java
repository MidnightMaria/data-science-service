package com.agnesmaria.datascience.controller;

import com.agnesmaria.datascience.service.DataFetchService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/integration")
public class IntegrationController {

    private final DataFetchService dataFetchService;

    // ✅ Manual constructor agar Spring bisa inject bean tanpa Lombok
    public IntegrationController(DataFetchService dataFetchService) {
        this.dataFetchService = dataFetchService;
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
}
