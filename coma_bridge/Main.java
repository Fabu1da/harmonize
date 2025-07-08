import de.wdilab.coma.integration.COMA_API;
import de.wdilab.coma.structure.MatchResult;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        // 1) Directories
        String srcPath = args.length > 0
            ? args[0]
            : "/Users/fabu1da/Desktop/schoolstuff/mastersThesis/Project/hamonize/assets/source";
        String trgPath = args.length > 1
            ? args[1]
            : "/Users/fabu1da/Desktop/schoolstuff/mastersThesis/Project/hamonize/assets/target/cvs";

        File sourceDir = new File(srcPath);
        File targetDir = new File(trgPath);
        if (!sourceDir.isDirectory() || !targetDir.isDirectory()) {
            System.err.println("Both arguments must be valid directories.");
            System.exit(1);
        }

        // 2) COMA++ setup
        COMA_API api = new COMA_API();
        String pattern = ".*\\.(owl|rdf|xsd|csv|sql|xml)$";
        File[] sourceFiles = sourceDir.listFiles((d, n) -> n.matches(pattern));
        File[] targetFiles = targetDir.listFiles((d, n) -> n.matches(pattern));

        // 3) Collect raw matches
        List<Map<String,Object>> raw = new ArrayList<>();

        // + runtimeByPair: record how long matchModelsDefault takes per file-pair
        Map<String, Double> runtimeByPair = new HashMap<>();

        if (sourceFiles != null && targetFiles != null) {
            for (File src : sourceFiles) {
                for (File trg : targetFiles) {
                    MatchResult mr = api.matchModelsDefault(
                        src.getAbsolutePath(), trg.getAbsolutePath()
                    );

                    // + start timer
                    long startNs = System.nanoTime();
                    
                    // + record elapsed time in seconds
                    double elapsed = (System.nanoTime() - startNs) / 1e9;
                    String pairKey = src.getName() + "→" + trg.getName();
                    runtimeByPair.put(pairKey, elapsed);


                    if (mr == null || mr.getMatchCount() == 0) continue;
                    for (Object s : mr.getSrcMatchObjects()) {
                        for (Object t : mr.getTrgMatchObjects(s)) {
                            float sim = mr.getSimilarity(s, t);
                            Map<String,Object> e = new HashMap<>();
                            e.put("src_file",   src.getName());
                            e.put("trg_file",   trg.getName());
                            e.put("source",     s.toString());
                            e.put("target",     t.toString());
                            e.put("similarity", sim);
                            // + include runtime on each entry (same for the whole pair)
                            e.put("runtime", elapsed);
                            raw.add(e);
                        }
                    }
                }
            }
        }

        // 4) Group by file‐pair key "src→trg"
        Map<String, List<Map<String,Object>>> byPair = new LinkedHashMap<>();
        for (Map<String,Object> e : raw) {
            String key = e.get("src_file") + "→" + e.get("trg_file");
            byPair.computeIfAbsent(key, k -> new ArrayList<>()).add(e);
        }

        // 5) Build JSON text
        StringBuilder out = new StringBuilder();
        out.append("[\n");
        boolean firstBlock = true;
        for (Map.Entry<String, List<Map<String,Object>>> kv : byPair.entrySet()) {
            if (!firstBlock) out.append(",\n");
            firstBlock = false;

            String[] parts = kv.getKey().split("→");
            String srcTbl = parts[0].replaceFirst("\\.[^.]+$", "");
            String trgTbl = parts[1].replaceFirst("\\.[^.]+$", "");

            // Pydantic fields
            boolean synthetic = false;                                  // COMA output is real data
            String genWith  = "coma_bridge/Main.java";                  // you can change this

            out.append("  {\n");
            out.append("    \"source_table\": \"").append(srcTbl).append("\",\n");
            out.append("    \"target_table\": \"").append(trgTbl).append("\",\n");
            out.append("    \"synthetic\": ").append(synthetic).append(",\n");
            out.append("    \"generated_with\": \"").append(genWith).append("\",\n");
            // + add the runtime (seconds) for this src→trg pair
            double rt = runtimeByPair.getOrDefault(kv.getKey(), 0.0);
            out.append("    \"runtime\": " + String.format(Locale.US, "%.3f", rt) + ",\n");
            out.append("    \"mappings\": [\n");

            List<Map<String,Object>> entries = kv.getValue();
            for (int i = 0; i < entries.size(); i++) {
                Map<String,Object> e = entries.get(i);
                // extract only column name after last '.'
                String fullSrc = (String)e.get("source");
                String fullTgt = (String)e.get("target");


                String colSrc = fullSrc.contains(".")
                    ? fullSrc.substring(fullSrc.lastIndexOf('.') + 1)
                    : fullSrc;
                String colTgt = fullTgt.contains(".")
                    ? fullTgt.substring(fullTgt.lastIndexOf('.') + 1)
                    : fullTgt;

                // fetch similarity
                float sim = (Float)e.get("similarity");
                String simStr = String.format(Locale.US, "%.4f", sim);

                out.append("      {")
                   .append("\"source_column\":\"").append(colSrc).append("\",")
                   .append("\"target_column\":\"").append(colTgt).append("\",")
                   .append("\"similarity\":").append(simStr)
                   .append("}");
                if (i < entries.size() - 1) out.append(",");
                out.append("\n");
            }

            out.append("    ]\n");
            out.append("  }");
        }
        out.append("\n]\n");

        // 6) Write to disk
        String jsonFile = "/Users/fabu1da/Desktop/schoolstuff/mastersThesis/Project/hamonize/assets/coma_result/pydantic_matches.json";
        try (FileWriter fw = new FileWriter(jsonFile)) {
            fw.write(out.toString());
            System.out.println("Wrote Pydantic JSON to " + jsonFile);
        } catch (IOException io) {
            System.err.println("Failed to write JSON: " + io.getMessage());
            System.exit(2);
        }
    }
}
