import React from 'react';
import Layout from '@theme/Layout';
import ChapterControls from '../components/ChapterControls';

// Custom layout for chapter pages that includes personalization controls
function ChapterLayout({children, ...props}) {
  return (
    <Layout {...props}>
      <div style={{ maxWidth: '100%', margin: '0 auto', padding: '0 2rem' }}>
        <ChapterControls />
        <div>
          {children}
        </div>
      </div>
    </Layout>
  );
}

export default ChapterLayout;