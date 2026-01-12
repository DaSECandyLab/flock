#include "flock/functions/scalar/scalar.hpp"
#include <iostream>

namespace flock {

nlohmann::json ScalarFunctionBase::Complete(nlohmann::json& columns, const std::string& user_prompt,
                                            ScalarFunctionType function_type, Model& model) {
    nlohmann::json data;
    // Simple Prompt Construction
    std::string prompt = user_prompt;
    nlohmann::json media_data = nlohmann::json::array();

    // Iterate over columns to replace placeholders
    for (const auto& column: columns) {
        if (!column.contains("name") || !column.contains("data") || column["data"].empty()) continue;

        std::string col_name = column["name"];
        std::string placeholder = "{{" + col_name + "}}";

        if (column.contains("type") && column["type"] == "image") {
            media_data.push_back(column);
            // For images, we might not replace text placeholders, or we replace with [IMAGE] or similar if needed.
            // But usually image data is passed separately to AddCompletionRequest.
            // Assuming user knows how to prompt with images or the backend handles "media_data".
        } else {
            // Basic string replacement for text data
            std::string val_str = column["data"][0].dump();// Get value as string (quoted if string)
            if (column["data"][0].is_string()) {
                val_str = column["data"][0].get<std::string>();
            }

            size_t pos = 0;
            bool place_flag = false;
            while ((pos = prompt.find(placeholder, pos)) != std::string::npos) {
                prompt.replace(pos, placeholder.length(), val_str);
                pos += val_str.length();
                place_flag = true;
            }

            if (!place_flag) {
                // We cannot find the placeholder, just push column name and value to the prompt back.
                prompt += col_name;
                prompt += ":";
                prompt += val_str;
                prompt += "; ";
            }
        }
    }

    OutputType output_type = OutputType::STRING;
    if (function_type == ScalarFunctionType::FILTER) {
        output_type = OutputType::BOOL;
    }

    model.AddCompletionRequest(prompt, 1, output_type, media_data);
    auto response = model.CollectCompletions();

    // Wrap the single response item in an array to match expected return signature (though now size is always 1)
    return nlohmann::json::array({response[0]});
};

nlohmann::json ScalarFunctionBase::BatchAndComplete(const nlohmann::json& tuples,
                                                    const std::string& user_prompt,
                                                    const ScalarFunctionType function_type, Model& model) {
    auto responses = nlohmann::json::array();

    // Check if there is data
    if (tuples.empty() || !tuples[0].contains("data") || tuples[0]["data"].empty()) {
        return responses;
    }

    int num_rows = static_cast<int>(tuples[0]["data"].size());

    for (int i = 0; i < num_rows; ++i) {
        // Construct single-row batch (columnar format but with 1 row)
        auto single_row_tuples = nlohmann::json::array();
        for (const auto& col: tuples) {
            nlohmann::json new_col = col;// Copy metadata
            new_col["data"] = nlohmann::json::array();
            new_col["data"].push_back(col["data"][i]);
            single_row_tuples.push_back(new_col);
        }

        try {
            auto response = Complete(single_row_tuples, user_prompt, function_type, model);

            // Allow for null/error handling if needed, but Complete throws or returns.
            if (!response.empty()) {
                responses.push_back(response[0]);
            } else {
                responses.push_back(nullptr);
            }
        } catch (const std::exception& e) {
            // Log or handle error? For now, maybe push null or rethrow?
            // To keep it simple and consistent with previous behavior of throwing on error:
            throw;
        }
    }

    return responses;
}

}// namespace flock
