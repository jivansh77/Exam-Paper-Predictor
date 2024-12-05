import React from 'react';

const AnalysisResults = ({ results }) => {
    return (
        <div>
            <h2>Analysis Results</h2>
            <pre>{JSON.stringify(results, null, 2)}</pre>
        </div>
    );
};

export default AnalysisResults; 