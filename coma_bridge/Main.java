
import de.wdilab.coma.integration.COMA_API;
import de.wdilab.coma.structure.MatchResult;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        // Optional: pass source/target dirs as args
        String srcPath = (args.length >= 1) ? args[0] : "/Users/fabu1da/Desktop/schoolstuff/mastersThesis/Project/hamonize/assets/source";
        String trgPath = (args.length >= 2) ? args[1] : "/Users/fabu1da/Desktop/schoolstuff/mastersThesis/Project/hamonize/assets/target/cvs";

        File sourceDir = new File(srcPath);
        File targetDir = new File(trgPath);



        if (!sourceDir.isDirectory() || !targetDir.isDirectory()) {
            System.err.println("Both arguments must be directories containing schema files.");
            System.exit(1);
        }

        // Initialize COMA++ API
        COMA_API api = new COMA_API();

        // Supported file extensions: owl, rdf, xsd, csv, sql, xml
        String pattern = ".*\\.(owl|rdf|xsd|csv|sql|xml)$";
        File[] sourceFiles = sourceDir.listFiles((d, n) -> n.matches(pattern));
        File[] targetFiles = targetDir.listFiles((d, n) -> n.matches(pattern));

        if (sourceFiles == null || targetFiles == null) {
            System.err.println("Error reading schema directories.");
            System.exit(2);
        }

        // Collect all matches as JSON-like objects
        List<Map<String, Object>> allMatches = new ArrayList<>();
        
        for (File src : sourceFiles) {
            for (File trg : targetFiles) {
                MatchResult result = api.matchModelsDefault(
                    src.getAbsolutePath(),
                    trg.getAbsolutePath()
                );
                if (result == null || result.getMatchCount() == 0) continue;

                for (Object srcEl : result.getSrcMatchObjects()) {
                    for (Object trgEl : result.getTrgMatchObjects(srcEl)) {
                        float sim = result.getSimilarity(srcEl, trgEl);
                        // Create JSON-like object
                        Map<String, Object> entry = new HashMap<>();
                        entry.put("source", srcEl.toString());
                        entry.put("target", trgEl.toString());
                        entry.put("similarity", sim);
                        entry.put("src_file", src.getName());
                        entry.put("trg_file", trg.getName());
                        allMatches.add(entry);
                    }
                }
            }
        }

       // Write to JSON file
        try (FileWriter fw = new FileWriter("/Users/fabu1da/Desktop/schoolstuff/mastersThesis/Project/hamonize/assets/output/matches.json")) {
            fw.write("[");
            for (int i = 0; i < allMatches.size(); i++) {
                Map<String, Object> m = allMatches.get(i);
                String jsonObj = String.format(
                    "{\"source\":\"%s\",\"target\":\"%s\",\"similarity\":%.4f,\"src_file\":\"%s\",\"trg_file\":\"%s\"}",
                    m.get("source"), m.get("target"), m.get("similarity"), m.get("src_file"), m.get("trg_file")
                );
                fw.write(jsonObj);
                if (i < allMatches.size() - 1) fw.write(",");
            }
            fw.write("]");
            System.out.println("Results written to matches.json");
        } catch (IOException e) {
            System.err.println("Error writing JSON file: " + e.getMessage());
            System.exit(3);
        }
    }
}

