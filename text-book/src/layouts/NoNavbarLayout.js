import React from 'react';
import clsx from 'clsx';
import Head from '@docusaurus/Head';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import SkipToContent from '@theme/SkipToContent';
import MDXProvider from '@theme/MDXComponents';
import {PageMetadata} from '@docusaurus/theme-common';
import {useKeyboardNavigation} from '@docusaurus/theme-common/internal';
import {useDocsSidebar} from '@docusaurus/theme-common/internal';
import {DocsSidebarProvider} from '@docusaurus/theme-common';
import LayoutProviders from '@theme/LayoutProviders';
import ErrorBoundary from '@docusaurus/ErrorBoundary';
import LayoutError from '@theme/LayoutError';

function NoNavbarLayout(props) {
  const {siteConfig, siteMetadata} = useDocusaurusContext();
  const {faviconUrl} = siteMetadata;
  const {children, noFooter, wrapperClassName, title, description} = props;

  useKeyboardNavigation();

  return (
    <LayoutProviders>
      <Head>
        {title && <title>{title}</title>}
        {description && <meta name="description" content={description} />}
        {description && <meta property="og:description" content={description} />}
        {faviconUrl && <link rel="shortcut icon" href={faviconUrl} />}
        <meta property="og:title" content={title} />
        <meta name="twitter:card" content="summary_large_image" />
      </Head>
      <PageMetadata title={title} description={description} />
      <SkipToContent />
      <div className={clsx('main-wrapper', wrapperClassName)}>
        <MDXProvider>{children}</MDXProvider>
      </div>
    </LayoutProviders>
  );
}

export default function NoNavbarWrapper(props) {
  const sidebar = useDocsSidebar();
  return sidebar ? (
    <DocsSidebarProvider sidebar={sidebar}>
      <NoNavbarLayout {...props} />
    </DocsSidebarProvider>
  ) : (
    <NoNavbarLayout {...props} />
  );
}