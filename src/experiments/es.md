PUT sr_boolq
{
  "settings": {
    "analysis": {
      "filter": {
        "wdg_preserve": {
          "type": "word_delimiter_graph",
          "preserve_original": true,
          "split_on_case_change": false
        }
      },
      "analyzer": {
        "scibasic": {
          "type": "custom",
          "tokenizer": "whitespace",
          "filter": [
            "lowercase",
            "asciifolding",
            "wdg_preserve"
          ]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "ref_id": { "type": "keyword" },
      "title": {
        "type": "text",
        "analyzer": "scibasic",
        "fields": {
          "exact": { "type": "keyword", "ignore_above": 256 }
        }
      },
      "abstract": { 
        "type": "text",
        "analyzer": "scibasic"
      }
    }
  }
}