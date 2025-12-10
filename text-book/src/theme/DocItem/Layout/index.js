import React from 'react';
import clsx from 'clsx';
import DocItemPaginator from '@theme/DocItem/Paginator';
import DocVersionBanner from '@theme/DocVersionBanner';
import DocItemFooter from '@theme/DocItem/Footer';
import DocItemContent from '@theme/DocItem/Content';
import DocBreadcrumbs from '@theme/DocBreadcrumbs';
import ChapterControls from '../../../components/ChapterControls';

import styles from './styles.module.css';

export default function DocItemLayout(props) {
  const {children} = props;

  return (
    <div className="row">
      <div className={clsx('col', styles.docItemCol)}>
        <DocVersionBanner />
        <DocBreadcrumbs />
        <div className="container">
          <ChapterControls />
          <DocItemContent>
            <article className={styles.docItemArticle}>
              {children}
            </article>
          </DocItemContent>
          <DocItemFooter />
        </div>
        <DocItemPaginator />
      </div>
    </div>
  );
}