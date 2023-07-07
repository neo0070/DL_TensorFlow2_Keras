/**
 * vitest 환경 설정 파일
 * 
 * @file vitest.config.json
 * @version 0.0.1
 * @license OBCon License 1.0
 * @copyright pnuskgh, All right reserved.
 * @author gye hyun james kim <pnuskgh@gmail.com> 
 */

import { defineConfig } from 'vitest/config'

export default defineConfig({
    test: {
        name: 'EFriendExpert',
        root: './packages',
        environment: 'node',
    }
})
