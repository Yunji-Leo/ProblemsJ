package actions;

public class BitTigerTest {
    BitTiger test = new BitTiger();
    BitTiger2 test2 = new BitTiger2();

    @org.junit.Test
    public void testMethod() {
        BitTiger.LRUCache lru = new BitTiger.LRUCache(1);
        lru.put(2, 1);
        lru.get(2);
        lru.put(3, 2);
        int result = test.largestRectangleArea(new int[]{2, 1, 2});
        test.minWindow("ADOBECODEBANC", "ABC");
        test.search(new int[]{3, 1}, 3);
    }

    @org.junit.Test
    public void testMethod2() {
        test2.countPrimes(1500000);
        test2.isHappy(19);
        test2.fractionToDecimal(1, 6);
    }
}
