setTimeout(function(){
    let domain = getQueryDomainWithGivenLanguage(isZhCNParam);
    CortanaApp.handleAnswerDomain(domain);
}, 500);

function getQueryDomainWithGivenLanguage(isZhCN) {
    let QueryDomain;
    (function(QueryDomain) {
     QueryDomain[QueryDomain["Unknown"] = 0] = "Unknown";
     QueryDomain[QueryDomain["ChitChat"] = 1] = "action://Client/AnswerDomain/Chitchat";
     QueryDomain[QueryDomain["Weather"] = 2] = "action://Client/AnswerDomain/Weather";
     QueryDomain[QueryDomain["Cat3A"] = 3] = "action://Client/AnswerDomain/Cat3a";
     QueryDomain[QueryDomain["Cat3B"] = 4] = "action://Client/AnswerDomain/Cat3b";
     })(QueryDomain || (QueryDomain = {}));
    
    function getAnswerDomain_zhCN() {
            let domain = QueryDomain.Unknown;
            let elContent = document.querySelector('#b_content');
            if (elContent) {
                let poleAnswer = elContent.querySelector('#b_pole');
                if (poleAnswer && poleAnswer.querySelector('ul[role]')) {
                    let feedSpan = poleAnswer.querySelector('ul[role]');
                    if (feedSpan.getAttribute('role').toLowerCase() === 'presentation') {
                        domain = QueryDomain.ChitChat;
                    }
                } else if (elContent.querySelector('.b_cat3a')) {
                    domain = QueryDomain.Cat3A;
                } else {
                    domain = QueryDomain.Cat3B;
                }
            }
            return QueryDomain[domain];
        }

        function getAnswerDomain_Non_zhCN() {
            let domain = QueryDomain.Unknown;
            let elContent = document.querySelector('#b_content');
            if (elContent) {
                if (elContent.querySelector('#b_pole')) {
                    if (elContent.querySelector('.wtr_core')) {
                        domain = QueryDomain.Weather;
                    } else {
                        domain = QueryDomain.ChitChat;
                    }
                } else if (elContent.querySelector('#b_result .b_cat3a')) {
                    domain = QueryDomain.Cat3A;
                } else {
                    domain = QueryDomain.Cat3B;
                }
            }
            return QueryDomain[domain];
        }

    return isZhCN == '1' ? getAnswerDomain_zhCN() : getAnswerDomain_Non_zhCN();
}
